import torch
import abc
import torch.nn.functional as F
import numpy as np
import einops

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def get_replacement_mapper_(x, y, max_len=77):

    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i) for i in inds_replace]
    inds_target = [get_word_inds(y, i) for i in inds_replace]

    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()

def get_replacement_mapper(prompts, max_len=77):
    mappers = []
    mapper = get_replacement_mapper_(prompts[0], prompts[1], max_len)
    mappers.append(mapper)
    return torch.stack(mappers)

def get_word_inds(text, word_place):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i+1 for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place + 1]
    return np.array(word_place)

def update_alpha_time_word(alpha, bounds, prompt_ind, word_inds=None):

    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])

    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])

    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, max_num_words=77):

    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

class LocalBlend:
    def __init__(self, prompts, words, device, threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold
        
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       

class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # print('attention store key ', key)
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach().cpu())

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    # prompts: list of prompts 
    # cross_replace_steps: {'default_': (start, end), 'word1': (start, end), 'word2': (start, end), ...}
    def __init__(self, prompts, opt, cross_replace_steps, self_replace_steps, local_blend):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, opt.t_gen, cross_replace_steps).to(opt.device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(opt.t_gen * self_replace_steps[0]), int(opt.t_gen * self_replace_steps[1])
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):

            h = attn.shape[0] // (self.batch_size)

            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    


class AttentionReplace(AttentionControlEdit):
    def __init__(self, prompts, opt, cross_replace_steps, self_replace_steps, local_blend=None):
        super(AttentionReplace, self).__init__(prompts, opt, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = get_replacement_mapper(prompts).to(opt.device)


    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      

        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = get_refinement_mapper(prompts)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps, equalizer,
                local_blend=None, controller=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


class DummyController:

    def __call__(self, *args):
        return args[0]

    def __init__(self):
        self.num_att_layers = 0    



def ca_forward(self, place_in_unet, controller=None):
    to_out = self.to_out
    if type(to_out) is torch.nn.modules.container.ModuleList:
        to_out = self.to_out[0]
    else:
        to_out = self.to_out

    def forward(x, context=None, mask=None):
        batch_size, sequence_length, dim = x.shape
        h = self.heads

        q = self.to_q(x)
        is_cross = context is not None
        context = context if is_cross else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = mask.reshape(batch_size, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = controller(attn, is_cross, place_in_unet)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return to_out(out)

    return forward

def register_recr(net_, count, place_in_unet, controller):

    if net_.__class__.__name__ == 'CrossAttention':
        net_.forward = ca_forward(net_, place_in_unet, controller)
        return count + 1
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            count = register_recr(net__, count, place_in_unet, controller)
    return count

def register_attention_control(diffusion_model, controller):

    if controller is None:
        controller = DummyController()

    cross_att_count = 0
    sub_nets = diffusion_model.named_children()
    for net in sub_nets:

        # print(net[1])
        if "input" in net[0]:

            cross_att_count += register_recr(net[1], 0, "down", controller)
        elif "output" in net[0]:

            cross_att_count += register_recr(net[1], 0, "up", controller)
        elif "mid" in net[0]:

            cross_att_count += register_recr(net[1], 0, "mid", controller)

    controller.num_att_layers = cross_att_count


