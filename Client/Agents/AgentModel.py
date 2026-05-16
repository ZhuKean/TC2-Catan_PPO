from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
from DeepLearning.CustomMaskablePPO import MaskablePPO
import logging


class BaseAgentModel(AgentRandom2):
    """
    The Agents here are not used for training, they are either used as opponents for training
    or for testing pretrained models
    """
    def __init__(self, name, seatNumber, model: MaskablePPO, playerTrading: bool=False, recordStats=False, jsettlersGame=False):

        super(BaseAgentModel, self).__init__(name, seatNumber, playerTrading=playerTrading, recordStats=recordStats, jsettlersGame=jsettlersGame)
        self.model                  = model
        self.jsettlersGame = jsettlersGame

    def getModelAction(self, game, possibleActions):
        """
        Uses model and env to get action
        """
        action_masks, indexActionDict = self.model.getActionMask(possibleActions)
        state = self.model.getObservation(game.gameState, self.seatNumber)
        action, _states = self.model.predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj
    
    def getRandomAction(self, game, possibleActions):
        """
        Used when model hasn't been trained on parts of game
        """
        randIndex = random.randint(0, len(possibleActions)-1)
        chosenAction = possibleActions[randIndex]
        if chosenAction.type == "MakeTradeOffer":
            self.tradeCount += 1
        return chosenAction


class AgentModel(BaseAgentModel):
    """
    Agent which uses passed model for all moves
    """
    def DoMove(self, game):

        # For JSettlers
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None
        
        possibleActions = self.GetPossibleActions(game.gameState)

        if self.jsettlersGame:
            print(f"POSSIBLE_ACTIONS: {self.resources[:5]}, DevCards: {self.developmentCards}")
            for action in possibleActions:
                if action:
                    print(f"     {action.type}")

        if len(possibleActions) == 1:
            actionObj = possibleActions[0]
        else:
            # Don't Allow to build roads when theres a possible settlement and we haven't built our 1st settlement
            if self.jsettlersGame and False:
                if game.gameState.currState[:5] != "START":
                    canBuildRoad = False
                    canBuyDevCard = False
                    for action in possibleActions:
                        if action.type == "BuildRoad":
                            canBuildRoad = True
                        elif action.type == "BuyDevelopmentCard":
                            canBuyDevCard = True
                    # Remove road option if haven't built 1st settlement or have longest road
                    if canBuildRoad:
                        if ((len(self.settlements) + len(self.cities) <= 2) and len(game.gameState.GetPossibleSettlements(self)) > 0) or (game.gameState.longestRoadPlayer == self.seatNumber):
                            possibleActions = [action for action in possibleActions if action.type != "BuildRoad"]
                            print("                 REMOVED BUILD ROAD OPTIONS")
                    # Remove buy dev card option if we haven't built a city
                    if canBuyDevCard:
                        if len(self.cities) == 0:
                            possibleActions = [action for action in possibleActions if action.type != "BuyDevelopmentCard"]
                            print("                 REMOVED BUY DEVCARD OPTIONS")
                    

            actionObj = self.getModelAction(game, possibleActions)

            if self.jsettlersGame:
                if actionObj.type == "MakeTradeOffer":
                    print(f"SELECTED_ACTION: {actionObj.type}, {actionObj.giveResources[:5]}_{actionObj.getResources[:5]}\n")
                else:
                    print(f"SELECTED_ACTION: {actionObj.type}\n")

            if self.playerTrading and actionObj.type == "MakeTradeOffer":
                self.tradeCount += 1
        
        if actionObj and actionObj.type == "EndTurn":
            self.playerTurns += 1

        return actionObj


class AgentMultiModel(BaseAgentModel):
    """
    Agent which uses a separate model for the setup phase and rest of game
    Must pass flag for whether setup is just settlements or settlements and roads
    If 'model' not passed will use random actions
    """

    def __init__(self, name, seatNumber,
                 setupModel, setup_obs_func, setup_mask_func,
                 gameplayModel, gameplay_obs_func, gameplay_mask_func,
                 fullSetup: bool, **kwargs):

        super().__init__(name, seatNumber, model=gameplayModel, **kwargs)
        self.setupModel = setupModel
        self.setup_obs_func = setup_obs_func
        self.setup_mask_func = setup_mask_func

        self.gameplay_obs_func = gameplay_obs_func
        self.gameplay_mask_func = gameplay_mask_func
        self.fullSetup = fullSetup

    def getSetupModelAction(self, game, possibleActions):
        # 使用开局专用的 Observation 和 Mask
        state = self.setup_obs_func(game.gameState, self.seatNumber)
        action_masks, indexActionDict = self.setup_mask_func(possibleActions)

        # 这里的 predict 使用 setupModel
        action, _ = self.setupModel.predict(state, action_masks=action_masks, deterministic=True)
        return indexActionDict[action.item()]

    def getModelAction(self, game, possibleActions):
        # 使用游戏阶段专用的 Observation 和 Mask (getObservationFull 等)
        state = self.gameplay_obs_func(game.gameState, self.seatNumber)
        action_masks, indexActionDict = self.gameplay_mask_func(possibleActions)

        # 这里的 predict 使用 self.model (即正在训练的 Gameplay 模型)
        action, _ = self.model.predict(state, action_masks=action_masks, deterministic=True)
        return indexActionDict[action.item()]

    def __getstate__(self):
        """告诉 pickle 哪些属性需要序列化，哪些不需要"""
        state = self.__dict__.copy()
        # 移除不可序列化的模型对象
        state['setupModel'] = None
        # 如果 self.model 也是神经网络模型，也需要移除
        if 'model' in state:
            state['model'] = None
        return state

    def __setstate__(self, state):
        """反序列化时恢复属性"""
        self.__dict__.update(state)
        # 注意：这里 setupModel 会变成 None。
        # 但没关系，因为 MCTS 在模拟“未来”时，P1 应该已经过了 Setup 阶段，
        # 或者在模拟中可以使用随机动作代替，这样就不会报错。

    def DoMove(self, game):
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if len(possibleActions) == 1:
            return possibleActions[0]
        else:
            # Setup stage
            if game.gameState.currState in ["START1A", "START2A"]:
                # Setup always using setupModel
                if self.setupModel is None:
                    actionObj = self.getRandomAction(game, possibleActions)
                    print("Warning: No Setup model. Random action taken.")
                else:
                    actionObj = self.getSetupModelAction(game, possibleActions)

            elif game.gameState.currState in ["START1B", "START2B"]:
                if self.fullSetup and self.setupModel is not None:
                    actionObj = self.getSetupModelAction(game, possibleActions)
                else:
                    actionObj = self.getRandomAction(game, possibleActions)

            # Downstream play
            elif self.model is None:
                actionObj = self.getRandomAction(game, possibleActions)
                print("Warning: No Downstream model. Random action taken.")

            elif hasattr(self.model, 'DoMove'):
                # if self.model is an Agent
                actionObj = self.model.DoMove(game)

            else:
                # else it's PPO model
                actionObj = self.getModelAction(game, possibleActions)


            if self.playerTrading and actionObj.type == "MakeTradeOffer":
                self.tradeCount += 1

            if actionObj.type == "EndTurn":
                self.playerTurns += 1

            return actionObj
    
