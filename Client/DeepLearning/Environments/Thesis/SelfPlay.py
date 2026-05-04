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
from DeepLearning.PPO import MaskablePPO
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
                reward += self.loseRewardAmount

        return None, reward, True, False, {}


class SelfPlayZKA(SelfPlayBase):

    def __init__(self, customBoard=None, players=None, trading=False, selfPlay=False, debug=False):
        super(SelfPlayZKA, self).__init__(customBoard=customBoard, players=players, trading=trading)
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Models/ZKA_model/model_6223354_elo332.zip')
        self.selfPlay = selfPlay

        # --- 修改部分：从 Episode 改为 Step ---
        # 设定在多少步之后完全停止 Dense Reward (例如 1000 局 * 平均 60 步 = 60000 步)
        self.dense_reward_end_step = 1000000
        self.current_step = 0
        # ------------------------------------

        self.debug = debug

        # 记录当前局内累积的 Dense Reward (用于末尾结算或观察)
        self.episode_dense_reward = 0

        # 博弈奖励参数
        self.score_diff_coeff = 1  # 分差系数
        # Reward settings
        self.winReward = True
        self.winRewardAmount = 30
        self.loseRewardAmount = -10
        self.vpActionReward = False  # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
        # Trading Rewards
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
        # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull

    def get_dense_weight(self):
        """基于总步数计算衰减权重"""
        if self.current_step >= self.dense_reward_end_step:
            return 0.0
        # 线性衰减：从 1.0 降到 0.0
        return 1.0 - (self.current_step / self.dense_reward_end_step)

    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0

        if self.debug:
            # 修改打印信息
            print("Total Steps: ", self.current_step)
            print("  Dense reward weight: ", self.get_dense_weight())

        # --- 修改部分：不再在 reset 里增加局数 ---
        # 局数计数已移除，改由 step 记录步数
        # ------------------------------------

        # Update opponents models if needed
        if self.selfPlay == True:
            if os.environ["UPDATE_MODELS_DIST"] == "True":
                modelName1 = os.environ["MODEL_1_NAME"]
                modelName2 = os.environ["MODEL_2_NAME"]
                modelName3 = os.environ["MODEL_3_NAME"]
                self.opponentModel1.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName1}")
                self.opponentModel2.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName2}")
                self.opponentModel3.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName3}")
                os.environ["UPDATE_MODELS_DIST"] = "False"
                if self.debug:
                    print(f"  Successfully using opponents: {modelName1}, {modelName2}, {modelName3}")

                self.game = CreateGame([AgentRandom2("P0", 0),
                                        AgentModel("P1", 1, self.opponentModel1),
                                        AgentModel("P2", 2, self.opponentModel2),
                                        AgentModel("P3", 3, self.opponentModel3)])
            else:
                self.game = CreateGame([
                    AgentRandom2("P0", 0),  # agent to be trained
                    AgentRandom2("P1", 1),
                    AgentRandom2("P2", 2),
                    AgentRandom2("P3", 3)
                ])
        else:
            self.game = CreateGame([
                AgentRandom2("P0", 0),  # agent to be trained
                AgentRandom2("P1", 1),
                AgentRandom2("P2", 2),
                AgentRandom2("P3", 3)
            ])

        if self.game is None:
            raise ValueError("CreateGame returned None.")

        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}

    def step(self, action):
        truncated = False
        done = False

        # --- 修改部分：在每一步执行时增加步数计数 ---
        self.current_step += 1
        # ----------------------------------------

        raw_dense_reward = 0

        # ==========================================
        # 1. 记录动作执行前的状态 (Before State)
        # ==========================================
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

        # ==========================================
        # 2. 执行动作 (Apply Action)
        # ==========================================
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        # ==========================================
        # 3. 记录动作执行后的状态并计算原始分 (After State & Raw Reward)
        # ==========================================
        if getattr(self, 'bankTradeReward', False):
            if actionObj.type == "BankTradeOffer":
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)

                if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                    raw_dense_reward += 1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    raw_dense_reward += 1 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    raw_dense_reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    raw_dense_reward += -1 * self.bankTradeRewardMultiplier
                if canBuildSettlementAfter == False and canBuildCityAfter == False and canBuildRoadAfter == False and canBuyDevCardAfter == False:
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

            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                raw_dense_reward += 10 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:
                raw_dense_reward += 10 * self.vpActionRewardMultiplier

        # ==========================================
        # 4. 应用步数衰减系数
        # ==========================================
        weighted_dense_reward = raw_dense_reward * self.get_dense_weight()

        if hasattr(self, 'episode_dense_reward'):
            self.episode_dense_reward += weighted_dense_reward

        # ==========================================
        # 5. 游戏结束判断与对手轮转
        # ==========================================
        if self.endCondition():
            return self.endGame(weighted_dense_reward)

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
            if agentAction:
                agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            if self.endCondition():
                return self.endGame(weighted_dense_reward)

        # ==========================================
        # 6. 准备当前 Agent 的下一步状态
        # ==========================================
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
        if winner_seat == 0:
            GAME_RESULTS.append(1)
            game_reward = self.winRewardAmount + (agent_score - avg_other_score) * self.score_diff_coeff
        else:
            GAME_RESULTS.append(0)
            game_reward = self.loseRewardAmount + (agent_score - avg_other_score) * self.score_diff_coeff

        total_reward = reward + game_reward
        if self.debug:
            print("  Final Reward:", total_reward)
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


  

class SelfPlayUniform(SelfPlayBase):
    """
    When threshold reached updated all opponents to current model. 
    """
    def __init__(self):
        super(SelfPlayUniform, self).__init__()

        # Load starting opponent model
        self.opponentModel = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_UNIFORM"] == "True":
            # Get name of model to update to
            modelName = os.environ["MODEL_NAME"]
            self.opponentModel.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Uniform/{modelName}")
            os.environ["UPDATE_MODELS_UNIFORM"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                                AgentModel("P1", 1, self.opponentModel),
                                AgentModel("P2", 2, self.opponentModel),
                                AgentModel("P3", 3, self.opponentModel)])
        # self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}


class SelfPlayDistribution(SelfPlayBase):
    """
    When threshold reached update oppenents to [most recent, random pick, random pick, random pick]
    """
    def __init__(self):
        super(SelfPlayDistribution, self).__init__()

        # Load starting opponent model
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_DIST"] == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName1}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName2}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName3}")
            os.environ["UPDATE_MODELS_DIST"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                                AgentModel("P1", 1, self.opponentModel1),
                                AgentModel("P2", 2, self.opponentModel2),
                                AgentModel("P3", 3, self.opponentModel3)])
        # self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}
    

class SelfPlayDense(SelfPlayBase):
    """
    When threshold reached update oppenents to [most recent, random pick, random pick, random pick]
    """
    def __init__(self):
        super(SelfPlayDense, self).__init__()

        # Load starting opponent model
        # self.opponentModel1 = MaskablePPO.load('DeepLearning/Thesis/6.DenseRewards/BaselineSelfPlay.zip')
        # self.opponentModel2 = MaskablePPO.load('DeepLearning/Thesis/6.DenseRewards/BaselineSelfPlay.zip')
        # self.opponentModel3 = MaskablePPO.load('DeepLearning/Thesis/6.DenseRewards/BaselineSelfPlay.zip')

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 100
        self.loseRewardAmount = -100
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Trading Rewards
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservationFull
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0


        # Update opponents models if needed
        if self.selfPlay == "True":
            if os.environ["UPDATE_MODELS_DIST"] == "True":
                modelName1 = os.environ["MODEL_1_NAME"]
                modelName2 = os.environ["MODEL_2_NAME"]
                modelName3 = os.environ["MODEL_3_NAME"]
                self.opponentModel1.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName1}")
                self.opponentModel2.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName2}")
                self.opponentModel3.set_parameters(f"DeepLearning/Models/ZKA_selfplay/{modelName3}")
                os.environ["UPDATE_MODELS_DIST"] = "False"

                self.game = CreateGame([AgentRandom2("P0", 0),
                                        AgentModel("P1", 1, self.opponentModel1),
                                        AgentModel("P2", 2, self.opponentModel2),
                                        AgentModel("P3", 3, self.opponentModel3)])
        else:
            self.game = CreateGame([
                AgentRandom2("P0", 0),  # agent to be trained
                AgentRandom2("P1", 1),
                AgentRandom2("P2", 2),
                AgentRandom2("P3", 3)
            ])

        # self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}
    
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
                reward += -5 * (10-self.agent.victoryPoints)

        return None, reward, True, False, {}

