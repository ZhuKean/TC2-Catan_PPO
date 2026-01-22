from DeepLearning.PPO import MaskablePPO

global_models = {
    # "Reward_win_10M": MaskablePPO.load("DeepLearning/Thesis/Rewards/Models/Reward_win/Reward_win_10M.zip"),
    # "SP_Distribution": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/Distribution/model_14966784.zip"),
    # "SP_Uniform": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/Uniform/model_14667776.zip"),
    # "VsModel": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/VsModel/model_1536000.zip"),
    "VsBaseline": MaskablePPO.load("DeepLearning/Models/Full/Full_vp_100k.zip"),
    "SelfPlayDense": MaskablePPO.load("DeepLearning/Models/SelfPlay/SelfPlay_SetupDotTotal_7vp_2M.zip"),
    "SetupSettlement": MaskablePPO.load("DeepLearning/Models/Tasks/FirstSettlement_50turns.zip"),
    "SetupCity": MaskablePPO.load("DeepLearning/Models/Trading_20Turns_CitySettlement/Trading_20Turns_CitySettlement_7M.zip")
}
