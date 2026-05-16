import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel, AgentMultiModel
from DeepLearning.GetActionMask import getActionMask, getActionMaskTrading
from DeepLearning.CustomMaskablePPO import MaskablePPO
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from DeepLearning.Thesis.Observations.get_observation_full import getObservationFull, lowerBound, upperBound

import numpy as np


class SelfPlayBase(CatanBaseEnv):

    def __init__(self, customBoard=None, players=None, trading=False, selfPlay=False):
        super(SelfPlayBase, self).__init__(customBoard=customBoard, players=players, trading=trading)

        self.selfPlay = selfPlay
        # Reward settings
        self.winReward = True
        self.winRewardAmount = 1
        self.loseRewardAmount = -1

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull

    def reset(self, seed=None):
        self.numTurns = 0
        return super(SelfPlayBase, self).reset()

    def endCondition(self) -> bool:
        if self.game.gameState.currState == "OVER":
            return True
        else:
            return False

    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn,
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        reward = 0

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        # Check if game Over
        if self.endCondition():
            return self.endGame(reward)

        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while self.selfPlay:
            # Only use model when right turn and more than 1 possible action
            if currPlayer.seatNumber == 0:
                possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                if len(possibleActions) > 1:
                    break
                elif possibleActions[0].type == "EndTurn":
                    self.numTurns += 1
                    self.agent.playerTurns += 1

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # Check if game Over
            if self.endCondition():
                return self.endGame(reward)

        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}

    def endGame(self, reward):
        wonGame = self.game.gameState.winner == 0
        if wonGame:
            GAME_RESULTS.append(1)
            if self.winReward:
                reward += self.winRewardAmount
        else:
            GAME_RESULTS.append(0)
            if self.winReward:
                reward += self.loseRewardAmount

        return None, reward, True, False, {}




class SelfPlayZKA(SelfPlayBase):
    def __init__(self, customBoard=None, players=None, trading=False, selfPlay=False, debug=False):
        super(SelfPlayZKA, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # 1. 加载对手模型 (请确保路径正确)
        try:
            default_model_path = 'DeepLearning/Models/ZKA_model/model_6223354_elo332.zip'
            self.opponentModel1 = MaskablePPO.load(default_model_path)
            self.opponentModel2 = MaskablePPO.load(default_model_path)
            self.opponentModel3 = MaskablePPO.load(default_model_path)
        except Exception as e:
            print(f"Warning: Opponent models not loaded: {e}")

        self.selfPlay = selfPlay
        self.debug = debug

        # 2. 衰减与步数计数器
        self.current_step = 0
        self.dense_reward_end_step = 1000000  # 100万步后 dense reward 降为 0

        # 3. 初始奖励设置 (对齐 SelfPlayDense)
        self.winRewardAmount = 50  # 对齐 Dense 版胜利奖励
        self.score_diff_coeff = 1  # ZKA 分差系数

        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
        self.denseRewards = True
        self.denseRewardMultiplier = 1
        self.vpActionRewardMultiplier = 1

        # 4. 环境空间定义
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull

    def get_dense_weight(self):
        """计算当前 Dense Reward 的权重 (1.0 -> 0.0)"""
        if self.current_step >= self.dense_reward_end_step:
            return 0.0
        return 1.0 - (self.current_step / self.dense_reward_end_step)

    def reset(self, seed=None):
        self.numTurns = 0
        if self.debug:
            print(f"Reset Game | Total Steps: {self.current_step} | Weight: {self.get_dense_weight():.2f}")

        # 自博弈模型更新逻辑
        if self.selfPlay:
            try:
                self.opponentModel1.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{os.environ['MODEL_1_NAME']}")
                self.opponentModel2.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{os.environ['MODEL_2_NAME']}")
                self.opponentModel3.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{os.environ['MODEL_3_NAME']}")
                os.environ["UPDATE_MODELS_DIST"] = "True"
            except:
                pass

        # 创建游戏 (确保 P0 是待训练 Agent)
        self.game = CreateGame([
            AgentRandom2("P0", 0),
            AgentModel("P1", 1, self.opponentModel1),
            AgentModel("P2", 2, self.opponentModel2),
            AgentModel("P3", 3, self.opponentModel3)
        ], customBoard=self.customBoard)

        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # 循环直到轮到 Agent (P0)
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        return self.getObservation(self.game.gameState), {}

    def step(self, action):
        self.current_step += 1
        raw_dense_reward = 0

        # 记录动作前状态
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        prevState = self.game.gameState.currState

        # 记录交易前能否负担（用于 BankTradeReward 逻辑）
        if self.bankTradeReward and prevState[:5] != "START":
            canBuildSettlementBefore = self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.CanAfford(BuildCityAction.cost)

        # 执行动作
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        # 计算 BankTrade 奖励
        if self.bankTradeReward and actionObj.type == "BankTradeOffer":
            canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
            if not canBuildSettlementBefore and canBuildSettlementAfter: raw_dense_reward += 1
            if not canBuildCityBefore and canBuildCityAfter: raw_dense_reward += 1

        # 计算基础 Dense Reward
        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                raw_dense_reward += 10
            elif actionObj.type == 'BuildCity':
                raw_dense_reward += 10
            elif actionObj.type == 'BuyDevelopmentCard':
                raw_dense_reward += 2
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                raw_dense_reward += 1

            if not biggestArmyBefore and self.agent.biggestArmy: raw_dense_reward += 10
            if not biggestRoadBefore and self.agent.biggestRoad: raw_dense_reward += 10

        # 应用当前权重的衰减
        weight = self.get_dense_weight()
        weighted_dense_reward = raw_dense_reward * weight

        # 检查游戏结束
        if self.endCondition():
            return self.endGame(weighted_dense_reward)

        # 推进对手回合
        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            if currPlayer.seatNumber == 0:
                possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                if len(possibleActions) > 1:
                    break
                elif possibleActions[0].type == "EndTurn":
                    self.numTurns += 1
                    self.agent.playerTurns += 1

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            if self.endCondition():
                return self.endGame(weighted_dense_reward)

        # 准备下一回合
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, weighted_dense_reward, False, False, {}

    def endGame(self, current_reward):
        """
        结算逻辑：
        前期 (weight=1): 100 / -5*(10-VP)
        后期 (weight=0): 100 + 分差 / -5*(10-VP) + 分差
        """
        weight = self.get_dense_weight()
        wonGame = self.game.gameState.winner == 0
        scores = [p.victoryPoints for p in self.game.gameState.players]
        agent_score = scores[0]
        avg_other_score = sum(scores[1:]) / 3

        # 1. 基础胜负奖励 (对齐 SelfPlayDense)
        if wonGame:
            GAME_RESULTS.append(1)
            base_game_reward = self.winRewardAmount  # +100
        else:
            GAME_RESULTS.append(0)
            base_game_reward = -5 * (10 - agent_score)  # -5 * 剩余分数

        # 2. ZKA 分差奖励 (随权重衰减而增强)
        # 这里的 (1 - weight) 确保了前期不触发分差，后期完全触发
        zka_diff_reward = (agent_score - avg_other_score) * self.score_diff_coeff
        final_game_reward = base_game_reward + (zka_diff_reward * (1 - weight))

        total_reward = current_reward + final_game_reward

        if self.debug:
            print(f"  Game Over | Agent VP: {agent_score} | Final Reward: {total_reward:.2f}")

        return None, total_reward, True, False, {}


class SelfPlayMultiModelZKA(SelfPlayBase):
    """
    专为 MultiModel 设计的 SelfPlay 环境。
    环境内部会自动使用 setupModel 完成所有玩家的开局放置。
    外部的 PPO 算法只会接收到 PLAY 阶段的 Observation，从而只训练 Gameplay 阶段。
    """

    def __init__(self, setupModel, setup_obs_func, setup_mask_func,
                 customBoard=None, players=None, trading=False, selfPlay=False, debug=False):
        super(SelfPlayMultiModelZKA, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # --- 存储 Setup 阶段的模型和特征函数 ---
        self.setupModel = setupModel
        self.setup_obs_func = setup_obs_func
        self.setup_mask_func = setup_mask_func

        # 加载对手 Gameplay 模型
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.selfPlay = selfPlay

        # 步数衰减机制
        self.dense_reward_end_step = 0
        self.current_step = 0
        self.debug = debug
        self.episode_dense_reward = 0
        self.score_diff_coeff = 1

        # 奖励设置
        self.winReward = True
        self.winRewardAmount = 30
        self.loseRewardAmount = -10
        self.vpActionReward = False
        self.vpActionRewardMultiplier = 1
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        # Gameplay 阶段的特征空间
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull

    def get_dense_weight(self):
        if self.current_step >= self.dense_reward_end_step:
            return 0.0
        return 1.0 - (self.current_step / self.dense_reward_end_step)

    def _create_multi_agent(self, name, seat, gameplay_model):
        """辅助工厂函数：创建 AgentMultiModel"""
        return AgentMultiModel(
            name=name,
            seatNumber=seat,
            setupModel=self.setupModel,
            setup_obs_func=self.setup_obs_func,
            setup_mask_func=self.setup_mask_func,
            gameplayModel=gameplay_model,
            gameplay_obs_func=self.getObservation,
            gameplay_mask_func=self.getActionMask,
            fullSetup=True
        )

    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0

        if self.debug:
            print("Total Steps: ", self.current_step)
            print("  Dense reward weight: ", self.get_dense_weight())

        # Update opponents models if needed
        if self.selfPlay == True and os.environ.get("UPDATE_MODELS_DIST") == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName1}")
            self.opponentModel2.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName2}")
            self.opponentModel3.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName3}")
            os.environ["UPDATE_MODELS_DIST"] = "False"
            if self.debug:
                print(f"  Successfully using opponents: {modelName1}, {modelName2}, {modelName3}")

        # ==========================================
        # 实例化所有玩家为 MultiModel
        # 注意：P0 的 gameplayModel 为 None，因为它的 gameplay 由外部 PPO 控制
        # ==========================================
        p0 = self._create_multi_agent("P0", 0, None)
        p1 = self._create_multi_agent("P1", 1, self.opponentModel1)
        p2 = self._create_multi_agent("P2", 2, self.opponentModel2)
        p3 = self._create_multi_agent("P3", 3, self.opponentModel3)

        self.game = CreateGame([p0, p1, p2, p3], customBoard=self.customBoard)

        if self.game is None:
            raise ValueError("CreateGame returned None.")

        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # ==========================================
        # 初始化循环：自动打完所有人的开局阶段
        # ==========================================
        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            # 如果轮到 P0，且已经脱离了开局阶段，就把控制权交给外部的 PPO！
            if currPlayer.seatNumber == 0 and self.game.gameState.currState not in ["START1A", "START1B", "START2A",
                                                                                    "START2B"]:
                break

            # 否则，无论是对手，还是处于开局阶段的 P0，都自动执行内部 DoMove
            agentAction = currPlayer.DoMove(self.game)
            if agentAction:
                agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # 到这里时，游戏必定处于 PLAY 状态，且轮到 P0 下棋
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)
        # ------------------ DEBUG 代码 ------------------
        if self.debug and self.current_step < 10:  # 只在前几步打印，防止刷屏
            print(f"\n--- DEBUG: Game {sum(GAME_RESULTS)} ---")
            for p in self.game.gameState.players:
                has_setup = p.setupModel is not None
                has_gameplay = p.model is not None
                print(f"[{p.name}] SetupModel: {has_setup}, GameplayModel: {has_gameplay}")
            print("----------------------------------\n")
        # ------------------------------------------------
        return observation, {}

    def step(self, action):
        truncated = False
        done = False
        self.current_step += 1
        raw_dense_reward = 0

        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        prevState = self.game.gameState.currState

        if getattr(self, 'bankTradeReward', False) and prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(0) and self.agent.CanAfford(
                BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.settlements and self.agent.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = self.game.gameState.GetPossibleRoads(self.agent) and self.agent.HavePiece(
                1) and self.agent.CanAfford(BuildRoadAction.cost)

        # 1. 外部 PPO 模型给出的动作执行
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        # 2. 奖励结算 (与你原来的逻辑完全一致)
        if getattr(self, 'bankTradeReward', False):
            if actionObj.type == "BankTradeOffer":
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)

                if not canBuildSettlementBefore and canBuildSettlementAfter: raw_dense_reward += 1 * self.bankTradeRewardMultiplier
                if not canBuildCityBefore and canBuildCityAfter: raw_dense_reward += 1 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore and not canBuildSettlementAfter: raw_dense_reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore and not canBuildCityAfter: raw_dense_reward += -1 * self.bankTradeRewardMultiplier
                if not canBuildSettlementAfter and not canBuildCityAfter and not canBuildRoadAfter and not canBuyDevCardAfter:
                    raw_dense_reward += -0.25 * self.bankTradeRewardMultiplier

        if getattr(self, 'denseRewards', True):
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                raw_dense_reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                raw_dense_reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                raw_dense_reward += 2 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                raw_dense_reward += 1 * self.denseRewardMultiplier

            if not biggestArmyBefore and self.agent.biggestArmy: raw_dense_reward += 10 * self.vpActionRewardMultiplier
            if not biggestRoadBefore and self.agent.biggestRoad: raw_dense_reward += 10 * self.vpActionRewardMultiplier

        weighted_dense_reward = raw_dense_reward * self.get_dense_weight()
        if hasattr(self, 'episode_dense_reward'):
            self.episode_dense_reward += weighted_dense_reward

        # 3. 推进游戏直到再次轮到 P0 做 Gameplay 决定
        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            if self.endCondition():
                return self.endGame(weighted_dense_reward)

            # ==========================================
            # 路由控制核心逻辑
            # ==========================================
            if currPlayer.seatNumber == 0:
                # 检查是否是在 Gameplay 阶段 (理论上 step 里不会触发 start，但写上更严谨)
                if self.game.gameState.currState not in ["START1A", "START1B", "START2A", "START2B"]:
                    possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                    if len(possibleActions) > 1:
                        break  # 将控制权交还给外部 PPO 算法
                    elif possibleActions[0].type == "EndTurn":
                        self.numTurns += 1
                        self.agent.playerTurns += 1
                # 如果是 P0 且处于特殊过渡状态，则继续往下执行 DoMove 让 MultiModel 自己处理

            agentAction = currPlayer.DoMove(self.game)
            if agentAction:
                agentAction.ApplyAction(self.game.gameState)

            currPlayer = self.players[self.game.gameState.currPlayer]

        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, weighted_dense_reward, done, truncated, {}

    def endGame(self, reward):
        scores = [p.victoryPoints for p in self.game.gameState.players]
        agent_score = scores[0]
        avg_other_score = sum(scores[1:]) / 3

        winner_seat = self.game.gameState.winner
        game_reward = 0

        # ==========================================
        # 补上下面这段代码，往计分板里写成绩！
        # ==========================================
        if winner_seat == 0:
            GAME_RESULTS.append(1)  # 我们赢了，记 1
        else:
            GAME_RESULTS.append(0)  # 对手赢了，记 0
        # ==========================================

        if winner_seat == 0:
            game_reward = self.winRewardAmount + (agent_score - avg_other_score) * self.score_diff_coeff
        else:
            game_reward = self.loseRewardAmount + (agent_score - avg_other_score) * self.score_diff_coeff

        total_reward = reward + game_reward
        if self.debug:
            print("  Final Reward:", total_reward)
        return None, total_reward, True, False, {}



    

class SelfPlayDense(SelfPlayBase):
    """
    SelfPlayDense 环境：支持自定义 trading 开启/禁用和 selfPlay 模式
    """

    def __init__(self, customBoard=None, players=None, trading=False, selfPlay=False, opponent_model_path=None):
        # 首先调用父类初始化
        super(SelfPlayDense, self).__init__(customBoard=customBoard, players=players, trading=trading,
                                            selfPlay=selfPlay)

        # 1. 校验逻辑：如果开启了 selfPlay 模式，必须提供模型路径
        if selfPlay:
            if opponent_model_path is None:
                raise ValueError("【错误】开启 selfPlay 模式时，必须通过 'opponent_model_path' 传入有效的模型路径！")

            # 检查文件物理路径是否存在
            import os
            if not os.path.exists(opponent_model_path):
                raise FileNotFoundError(f"【错误】指定的模型文件不存在: {os.path.abspath(opponent_model_path)}")

        # 2. 赋值路径属性
        self.base_model_path = opponent_model_path

        # 3. 加载对手模型
        # 注意：只有在 selfPlay 为 True 且路径有效时才加载，节省内存和加载时间
        if selfPlay and self.base_model_path:
            print(f"正在从 {self.base_model_path} 加载对手模型...")
            self.opponentModel1 = MaskablePPO.load(self.base_model_path)
            self.opponentModel2 = MaskablePPO.load(self.base_model_path)
            self.opponentModel3 = MaskablePPO.load(self.base_model_path)
        else:
            # 非 Self-Play 模式或未传入路径时，初始化为 None，防止后面逻辑引用报错
            self.opponentModel1 = None
            self.opponentModel2 = None
            self.opponentModel3 = None

        self.selfPlay = selfPlay

        # 奖励设置
        self.winReward = True
        self.winRewardAmount = 50
        self.loseRewardCoefficient = 5  # 失败时的分差惩罚系数 (对齐 SelfPlayZKA)
        self.bankTradeRewardMultiplier = 1
        self.vpActionReward = False
        self.vpActionRewardMultiplier = 1
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull

    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # 2. 自博弈模型更新逻辑 (由外部 PPO 脚本通过环境变量触发)
        if self.selfPlay and os.environ["UPDATE_MODELS_DIST"] == "False":
            try:
                m1 = os.environ.get('MODEL_1_NAME')
                m2 = os.environ.get('MODEL_2_NAME')
                m3 = os.environ.get('MODEL_3_NAME')

                # 更新权重
                self.opponentModel1.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{m1}")
                self.opponentModel2.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{m2}")
                self.opponentModel3.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{m3}")

                # 打印检查信息
                print(f"\n[Reset] 成功同步对手模型权重: P1={m1}, P2={m2}, P3={m3}")
                os.environ["UPDATE_MODELS_DIST"] = "True"

            except Exception as e:
                print(f"[Reset] 模型同步失败: {e}")

        # 3. 构建对手 Agent 列表
        if self.selfPlay:
            opponent_agents = [
                AgentModel("P1", 1, self.opponentModel1),
                AgentModel("P2", 2, self.opponentModel2),
                AgentModel("P3", 3, self.opponentModel3)
            ]
        else:
            # 基础训练模式：全是随机对手
            opponent_agents = [AgentRandom2(f"P{i}", i) for i in range(1, 4)]

        # 4. 创建游戏 (P0 是当前正在训练的学习者)
        self.game = CreateGame([AgentRandom2("P0", 0)] + opponent_agents, customBoard=self.customBoard)
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # 5. 循环直到轮到 P0 的回合
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # 返回初始观测和掩码
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        return self.getObservation(self.game.gameState), {}
    
    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        reward = 0
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        vpDevCardBefore = self.agent.developmentCards[VICTORY_POINT_CARD_INDEX]
        prevState = self.game.gameState.currState

        if self.bankTradeReward and prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(g_pieces.index('SETTLEMENTS')) and self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.settlements and self.agent.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = self.game.gameState.GetPossibleRoads(self.agent) and self.agent.HavePiece(g_pieces.index('ROADS')) and self.agent.CanAfford(BuildRoadAction.cost)

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        if self.bankTradeReward:
            if actionObj.type == "BankTradeOffer":
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
                if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildSettlementAfter == False and canBuildCityAfter == False and canBuildRoadAfter == False and canBuyDevCardAfter == False:
                    reward += -0.25 * self.bankTradeRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 2 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            # Using dev card
            # elif actionObj.type[:3] == 'Use':
            #     reward += 1
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 10 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 10 * self.vpActionRewardMultiplier

        # Check if game Over
        if self.endCondition():
            return self.endGame(reward)
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            # Only use model when right turn and more than 1 possible action
            if currPlayer.seatNumber == 0:
                possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                if len(possibleActions) > 1:
                    break
                elif possibleActions[0].type == "EndTurn":
                    self.numTurns += 1
                    self.agent.playerTurns += 1

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # Check if game Over
            if self.endCondition():
                return self.endGame(reward)
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}
    
    def endGame(self, reward):
        wonGame = self.game.gameState.winner == 0
        if wonGame:
            GAME_RESULTS.append(1)
            if self.winReward:
                reward += self.winRewardAmount
        else:
            GAME_RESULTS.append(0)
            if self.winReward:
                reward += -self.loseRewardCoefficient * (10-self.agent.victoryPoints)

        return None, reward, True, False, {}

