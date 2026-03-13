import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel
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

    def __init__(self, customBoard=None, players=None, trading=False, selfPlay=False, total_episodes=10000):
        super(SelfPlayZKA, self).__init__(customBoard=customBoard, players=players, trading=trading)
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Models/ZKA_model/Better_Densereward.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Models/ZKA_model/Better_Densereward.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Models/ZKA_model/Better_Densereward.zip')
        self.selfPlay = selfPlay
        # 设定在多少局之后完全停止 Dense Reward
        self.dense_reward_end_episode = 100000
        self.current_episode = 0

        # 记录当前局内累积的 Dense Reward (用于末尾结算或观察)
        self.episode_dense_reward = 0

        # 博弈奖励参数
        self.win_base_reward = 3.0
        self.lose_base_reward = -1.0
        self.score_diff_coeff = 0.2  # 分差系数
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

    def get_dense_weight(self):
        """基于局数计算衰减权重"""
        if self.current_episode >= self.dense_reward_end_episode:
            return 0.0
        # 线性衰减：从 1.0 降到 0.0
        return 1.0 - (self.current_episode / self.dense_reward_end_episode)

    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # 增加局数计数，用于奖励衰减逻辑
        if hasattr(self, 'current_episode'):
            self.current_episode += 1

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
                print(f"Using opponents: {modelName1}, {modelName2}, {modelName3}")

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
            raise ValueError("CreateGame returned None. Check if AgentModel/AgentRandom2 initialization is correct.")
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
        truncated = False
        done = False

        # 用来记录当前的原始基础奖励
        raw_dense_reward = 0

        # ==========================================
        # 1. 记录动作执行前的状态 (Before State)
        # ==========================================
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        vpDevCardBefore = self.agent.developmentCards[2]  # 假设 VICTORY_POINT_CARD_INDEX 为 2，请根据你的代码调整
        prevState = self.game.gameState.currState

        if getattr(self, 'bankTradeReward', False) and prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(0) and self.agent.CanAfford(
                BuildSettlementAction.cost)  # 假设 g_pieces 对应索引，请按需替换回原变量
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

                # Trades which allow us to build
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

        # 如果还在 Dense Reward 阶段，计算行为分 (这里建议保持 self.denseRewards 为 True，通过权重来衰减)
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
        # 4. 应用局数衰减系数
        # ==========================================
        weighted_dense_reward = raw_dense_reward * getattr(self, 'get_dense_weight', lambda: 1.0)()

        if hasattr(self, 'episode_dense_reward'):
            self.episode_dense_reward += weighted_dense_reward

        # ==========================================
        # 5. 游戏结束判断与对手轮转
        # ==========================================
        if self.endCondition():
            return self.endGame(weighted_dense_reward)

        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            # 轮到 Agent (Seat 0) 的回合
            if currPlayer.seatNumber == 0:
                possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                if len(possibleActions) > 1:
                    break
                elif possibleActions[0].type == "EndTurn":
                    self.numTurns += 1
                    self.agent.playerTurns += 1

            # 对手执行动作
            agentAction = currPlayer.DoMove(self.game)
            if agentAction:  # 安全校验，防止 DoMove 返回 None 导致崩溃
                agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # 如果对手的动作导致游戏结束
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
        """
        结算博弈奖励：结合 3:-1 逻辑与得分差
        """
        scores = [p.victoryPoints for p in self.game.gameState.players]
        agent_score = scores[0]
        other_scores = scores[1:]
        avg_other_score = sum(other_scores) / len(other_scores)

        winner_seat = self.game.gameState.winner

        # --- 4. 计算博弈奖励 (Game Reward) ---
        game_reward = 0
        if winner_seat == 0:
            GAME_RESULTS.append(1)
            # 赢家基础奖 + 领先分差奖
            game_reward = self.winRewardAmount + (agent_score - avg_other_score) * 0.5
        else:
            GAME_RESULTS.append(0)
            # 输家基础罚 + 落后分差罚
            game_reward = self.loseRewardAmount + (agent_score - avg_other_score) * 0.5

        # 最终奖励 = 累积的衰减 Dense Reward + 一次性结算的博弈奖励
        total_reward = reward + game_reward

        return None, total_reward, True, False, {}

    def calculate_legacy_dense_reward(self, action):
        """这里放你之前产生 150 分左右奖励的旧逻辑"""
        reward = 0
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        vpDevCardBefore = self.agent.developmentCards[VICTORY_POINT_CARD_INDEX]
        prevState = self.game.gameState.currState

        if self.bankTradeReward and prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(
                g_pieces.index('SETTLEMENTS')) and self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.settlements and self.agent.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = self.game.gameState.GetPossibleRoads(self.agent) and self.agent.HavePiece(
                g_pieces.index('ROADS')) and self.agent.CanAfford(BuildRoadAction.cost)

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
        return reward


  

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

