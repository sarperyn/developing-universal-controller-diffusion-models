import os

##############################################################################################################################
"""
Change those paths according to your local machine you use for experiments.
"""
MACHINE = "gursey-sarper"

if MACHINE == "gursey-ozgur":
    CWD = "/home/okara7/codes/stable-diff-asyrp"
    DATA_PATH = "/data/datasets/"
    CHAIR_PATH = "/data/datasets/CHAIR/chair_fixed_elev/seen/"
    CHAIR_PATH2 = "/data/datasets/CHAIR/chair_varying_elev/seen/"
    PRETRAINED_PATH = "/data/pretrained_models/v2-1_512-ema-pruned.ckpt"
    CONFIG_SD_PATH = "/home/okara7/codes2/sd-asyrp/configs/v2-inference.yaml"
    DEPTH_CONFIG_SD_PATH = "/home/okara7/codes2/sd-asyrp/configs/v2-midas-inference.yaml"
    DEPTH_SD_PATH = "/data/pretrained_models/512-depth-ema.ckpt"
    MIDAS_SD_PATH =  "/data/pretrained_models/dpt_hybrid-midas-501f0c75.pt"


elif MACHINE == "gursey-sarper":
    CWD = "/home/syurtseven7/codes/stable-diff-rotation"
    DATA_PATH = "/data/datasets/"
    CHAIR_PATH = "/data/datasets/CHAIR/chair_fixed_elev/seen/"
    CHAIR_PATH2 = "/data/datasets/CHAIR/chair_varying_elev/seen/"
    COMPLEX_PATH = "/data/0912_chairs/"
    PRETRAINED_PATH = "/data/pretrained_models/v2-1_512-ema-pruned.ckpt"
    CONFIG_SD_PATH = "/home/syurtseven7/codes/stable-diff-rotation/configs/v2-inference.yaml"
    DEPTH_CONFIG_SD_PATH = "/home/syurtseven7/codes/stable-diff-rotation/configs/v2-midas-inference.yaml"
    DEPTH_SD_PATH = "/data/pretrained_models/512-depth-ema.ckpt"
    MIDAS_SD_PATH =  "/data/pretrained_models/dpt_hybrid-midas-501f0c75.pt"


    

##############################################################################################################################
CELEBA_DATASET_IMGS = os.path.join(DATA_PATH, "CELEBA", "celeba", "img_align_celeba")
PROTOTOYPE_TEST_SET = os.path.join(DATA_PATH, "prototype_test_set")
##############################################################################################################################
DIFFUSION_PRETRAINED_MODEL_PATHS = {
    "FFHQ_P2": os.path.join(
        PRETRAINED_PATH, "diffusion_p2-weighted_pretrained/ffhq_p2.pt"
    ),
    "CelebA_HQ_P2": os.path.join(
        PRETRAINED_PATH, "diffusion_p2-weighted_pretrained/celebahq_p2.pt"
    ),
    "Chair-WB_i-DDPM_64": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_64/model120000_id_64.pt"
    ),
    "Chair-WB_i-DDPM_128": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_128/model300000_id_128.pt"
    ),
    "Chair-WB_GD_64": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_64/model070000_gd_64.pt"
    ),
    "Chair-WB_GD_128": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_128/model378000_gd_128.pt"
    ),
    "Chair-WB_EGD_128": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_128/ema_model378000_gd_128.pt"
    ),
    "Chair-WB_GD_ResUpDown_128": os.path.join(
        PRETRAINED_PATH,
        "chair_diffusion/img-size_128/img-size_128417000_resupdown-True.pt",
    ),
    "Chair-WB_GD_ResUpDown_128_N": os.path.join(
        PRETRAINED_PATH, "chair_diffusion/img-size_128/model261000_resupdown-True.pt"
    ),
}

CELEBA_PRETRAINED_MODEL_PATHS = {
    "Binary_Classifiers": os.path.join(PRETRAINED_PATH, "celeba_classifiers")
}
##############################################################################################################################
MODEL_CONFIG = {
    "Chair-WB_i-DDPM_64": {"img_size": (3, 64, 64), "learned_sigma": False},
    "FFHQ_P2": {"img_size": (3, 256, 256), "learned_sigma": True},
    "CelebA_HQ_P2": {"img_size": (3, 256, 256), "learned_sigma": True},
    "Chair-WB_i-DDPM_128": {"img_size": (3, 128, 128), "learned_sigma": False},
    "Chair-WB_GD_64": {"img_size": (3, 64, 64), "learned_sigma": False},
    "Chair-WB_GD_128": {"img_size": (3, 128, 128), "learned_sigma": False},
    "Chair-WB_EGD_128": {"img_size": (3, 128, 128), "learned_sigma": False},
    "Chair-WB_GD_ResUpDown_128": {"img_size": (3, 128, 128), "learned_sigma": True},
    "Chair-WB_GD_ResUpDown_128_N": {"img_size": (3, 128, 128), "learned_sigma": True},
}

BETA_CONFIG = {
    "Chair-WB_GD_128": {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "total_timesteps": 1000,
    },
    "FFHQ_P2": {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "total_timesteps": 1000,
    },
    "CelebA_HQ_P2": {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "total_timesteps": 1000,
    },
}

##############################################################################################################################

CELEBA_ATTRS = "5_o_Clxock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(
    " "
)

EXCLUDED_CHAIRS = [
    "03001627_1157d8d6995da5c0290d57214c8512a4",
    "03001627_1007e20d5e811b308351982a6e40cf41",
    "03001627_1006be65e7bc937e9141f9b58470d646",
    "03001627_249f3eb6a7236ff7593ebeeedbff73b",
    "03001627_26aa22bd1da8b8c5b1a5c6ecbc81953c",
    "03001627_26a6ce644504c5fa22963ea1e168015d",
    "03001627_20fd21021f3c8e5fcce6278f5ffb13a",
    "03001627_20fbab2b8770a1cbf51f77a6d7299806",
    "03001627_18fd8342fa5d1d4f5268b70948af88b2",
    "03001627_13bc03eca6aad4b2d7bf6fb68df7f786",
    "03001627_2e0beb3b6927a2b7e45ef4135c266a12",
    "03001627_2e0b6f6d19078424c3bd24f986301745",
    "03001627_2c548222017955df4530ae9f1281950f",
    "03001627_2a2d705d0238396488422a4c20c0b1e6",
    "03001627_1b67a3a1101a9acb905477d2a8504646",
    "03001627_1b6c268811e1724ead75d368738e0b47",
    "03001627_1a8bbf2994788e2743e99e0cae970928",
    "03001627_1aa07508b731af79814e2be0234da26c",
    "03001627_1ab8a3b55c14a7b27eaeab1f0c9120b7",
    "03001627_1ad766f9e95ce308aa425ecb668e59de",
    "03001627_1aeb17f89e1bea954c6deb9ede0648df",
    "03001627_1b5e876f3559c231532a8e162f399205",  #
    "03001627_1b7ba5484399d36bc5e50b867ca2d0b9",
    "03001627_1b7bef12c554c1244c686b8271245d1b",
    "03001627_1b8e84935fdc3ec82be289de70e8db31",
    "03001627_1b67a3a1101a9acb905477d2a8504646",
    "03001627_1b80175cc081f3e44e4975e87c20ce53",
    "03001627_1b92525f3945f486fe24b6f1cb4a9319",
    "03001627_1bb81d54471d7c1df51f77a6d7299806",
    "03001627_1be38f2624022098f71e06115e9c3b3e",
    "03001627_1be0108997e6aba5349bb1cbbf9a4206",
    "03001627_1c2caacac14dfa0019fb4103277a6b93",
    "03001627_1c5d66f3e7352cbf310af74324aae27f",
    "03001627_1c45b266d3c879dab36dcc661f3905d",
    "03001627_1c173d970e21e9a8be95ff480950e9ef",
    "03001627_1c199ef7e43188887215a1e3ffbff428",
    "03001627_1c685bc2a93f87a2504721639e19f609",
    "03001627_1c758127bc4fdb18be27e423fd45ffe7",
    "03001627_1cad298ed14e60f866e6ad37fee011e",
    "03001627_1cc6f2ed3d684fa245f213b8994b4a04",
    "03001627_1cd152cfd71cd314e2798a633e84d70b",
    "03001627_1d1b37ce6d72d7855096c0dd2594842a",
    "03001627_1d6f4020cab4ec1962d6a66a1a314d66",
    "03001627_1d6faeb6d77d1f2cf95cd8df6bebbc3a",
    "03001627_1d37a7fbe0810f963e83b2d32ed5f665",
    "03001627_1d99f74a7903b34bd56bda2fb2008f9d",
    "03001627_1d498876c8890f7786470a1318504fef",
    "03001627_1d1641362ad5a34ac3bd24f986301745",
    "03001627_1e1b70bdfafb9d22df2fa7eaa812363c",
    "03001627_1e2ddaef401676915a7934ad3293bab5",
    "03001627_1e6cfd4bfc6270f822b5697e1c26fdf8",
    "03001627_1e7bc7fd20c61944f51f77a6d7299806",
    "03001627_1e15f238da6b4bc546b9f3b6ee20ff4b",
    "03001627_1e44e3c128b667b4fdef1c01cbd4ae0c",
    "03001627_1e1151a459002e85508f812891696df0",
    "03001627_1eb2e372a204a61153baab6c8235f5db",
    "03001627_1eb5613aa22df87b8ef9327f5d5c024d",
    "03001627_1eb8558d0f6885a1268ecf2838ad6f15",
    "03001627_1eeae84f10df85cd74984b9cd0997a52",
    "03001627_1eed5ebb2af372cb5438b83aba42ca46",
    "03001627_1ef31b046039bf985c8a41baad250b1b",
    "03001627_10a1783f635b3fc181dff5c2e57ad46e",
    "03001627_11a06e6f68b1d99c8687ff9b0b4e4ac",
    "03001627_11c8f43ef796e23941e621b1a4bf507f",
    "03001627_11c9c57efad0b5ec297936c81e7f6629",
    "03001627_11d4f2a09184ec972b9f810ad7f5cbd2",
    "03001627_11d9817e65d7ead6b87028a4b477349f",
    "03001627_11dba3a8d7b7210f5ff61a3a2a0e2484",
    "03001627_11e521e41ff6a64922e4620665c23c97",
    "03001627_11e28120789c20abc8687ff9b0b4e4ac",
    "03001627_11f1511799d033ff7962150cab9888d6",
    "03001627_11fa9b044482814ef91663a74ccd2338",
    "03001627_12d7ca3c06faaadf17b431cae0dd70ed",
    "03001627_12f395270a3316d01666e1246e760f82",
    "03001627_13d4fceabfda96c0bff8d8db0f9298ac",
    "03001627_13fdf00cde077f562f6f52615fb75fca",
    "03001627_17b7a0e3c70dbc3d90a6b1b2b5522960",
    "03001627_17b558e72a4d76ef8517036a5ca6b1c7",
    "03001627_17d4c0f1b707e6dd19fb4103277a6b93",
    "03001627_17d7a3e8badbd881fceff3d071111703",
    "03001627_17e916fc863540ee3def89b32cef8e45",
    "03001627_18d391ede29e2edb990561fc34164364",
    "03001627_18e5d3054fba58bf6e30a0dcfb43d654",
    "03001627_19c01531fed8ae0b260103f81999a1e1",
    "03001627_19cbb7fd3ba9324489a921f93b6641da",
    "03001627_19dbb6cbd039caf0a419f44cba56de45",
    "03001627_19ff1d5665c1a68677b8fc2abf845259",
    "03001627_103a0a413d4c3353a723872ad91e4ed1",
    "03001627_103a60f3b09107df2da1314e036b435e",
    "03001627_103c31671f8c0b1467bb14b25f99796e",
    "03001627_103d77d63f0d68a044e6721e9db29c1b",
    "03001627_107ed94869ed6f1be13496cd332ce78f",
    "03001627_114f72b38dcabdf0823f29d871e57676",
    "03001627_116a9cd5ac9a008ee8cb931b01d0a372",
    "03001627_117bd6da01905949a81116f5456ee312",
    "03001627_117c0e0aafc0c3f81015cdff13e6d9f3",
    "03001627_124ef426dfa0aa38ff6069724068a578",
    "03001627_126e65c55961e5c166f17d3ad78f5a62",
    "03001627_131edf0948b60ee6372c8cd7d07d8ddc",
    "03001627_179b88264e7f96468b442b160bcfb7fd",
    "03001627_181b65afaeca2ee1a6536c847a708e24",
    "03001627_195c379defdff8b81dff5c2e57ad46e",
    "03001627_197ae965385b8187ae663e348bd216d3",
    "03001627_198bd40d94930950df6cfab91d65bb91",
    "03001627_1006be65e7bc937e9141f9b58470d646",
    "03001627_1007e20d5e811b308351982a6e40cf41",
    "03001627_1013f70851210a618f2e765c4a8ed3d",
    "03001627_1016f4debe988507589aae130c1f06fb",
    "03001627_1033ee86cc8bac4390962e4fb7072b86",
    "03001627_1055f78d441d170c4f3443b22038d340",
    "03001627_1063d4fcd366de4060e37b3f76995f8b",
    "03001627_1093d35c2ac73bb74ca84d60642ec7e8",
    "03001627_1166b15756ed9b8897317616969032",
    "03001627_1190af00b6c86c99c3bd24f986301745",
    "03001627_1230d31e3a6cbf309cd431573238602d",
    "03001627_1762c260d358a1751b17743c18fb63dc",
    "03001627_1769c3cf3391d5c1a1d7c136d0e341",
    "03001627_11347c7e8bc5881775907ca70d2973a4",
    "03001627_11358c94662a68117e66b3e5c11f24d4",
    "03001627_11506b96d41f7d3dd7c4a943f33e0384",
    "03001627_13076ebf8b5cc457b8d6f69a14683de3",
    "03001627_19666f52289092a3394a3bbfc81460",
    "03001627_120735afde493c277ff6ace05b36a5",
    "03001627_123305d8ccc0dc6346918a1d9c256af3",
    "03001627_124117cdec71699850c2ec40da48fd9d",
    "03001627_128517f2992b6fb92057e1ae1c4cc928",
    "03001627_187222bc1f81d57b781d9dcb8ecbccc",
    "03001627_191360ba29f3d296ff458e602ebccbb0",
    "03001627_1820138eca42749262e4024c69de065d",
    "03001627_1937193cf5079b623eec26c23f5bc80b",
    "03001627_17352867f5661212c8687ff9b0b4e4ac",
    "03001627_19319101e064eb1019fb4103277a6b93",
    "03001627_113016635d554d5171fb733891076ecf",
    "03001627_18005751014e6ee9747c474f2e537e26",
    "03001627_1803116582841b39a8ecfcf20e8cc0a",
]
