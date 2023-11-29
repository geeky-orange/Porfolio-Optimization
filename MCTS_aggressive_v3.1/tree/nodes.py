import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from util.portfolio_property import get_risk_profile
risk_profile = get_risk_profile()


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, asset_list, parent=None):
        """
        Parameters
        ----------
        asset_list : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.asset_list = asset_list
        self.parent = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """
        Returns
        -------
        list of mctspy.games.common.AbstractGameAction
        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        temp = self.untried_actions
        # print(temp)
        return len(temp) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(0, len(possible_moves))]


class PortfolioConstructionTreeSearchNode(MonteCarloTreeSearchNode):
    def __init__(self, asset_list, parent=None):
        super().__init__(asset_list, parent)
        self._number_of_visits = 0.
        self._results = {}
        self._untried_actions = None
        self._portfolios_info = None

    @property
    def untried_actions(self):
        # print("_untried_actions", self._untried_actions)
        if self._untried_actions is None:
            self._untried_actions = self.asset_list.get_possible_asset()
        return self._untried_actions

    @property
    def q(self):
        if risk_profile not in self._results.keys():
            return 0
        return self._results[risk_profile]

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        """
        expand the current node to get new asset list
        and add the next node to the child node
        :return:
        """
        action = self.untried_actions.pop(0)
        next_asset = self.asset_list.move(action)
        # print(next_asset.selected_asset_list)
        child_node = PortfolioConstructionTreeSearchNode(
            next_asset, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        """
        :return: if the node is terminal node
        """
        return self.asset_list.is_satisfied()

    def rollout(self):
        """
        do rollout to get the result: None (continue), 'conservative' (stop),
        'aggressive (stop)', 'fail' (stop), ...
        :return:
        """
        current_rollout_asset = self.asset_list
        """while not current_rollout_asset.is_satisfied():  # /? change the logic to reduce running time
            possible_asset = current_rollout_asset.get_possible_asset()
            asset = self.rollout_policy(possible_asset)
            current_rollout_asset = current_rollout_asset.move(asset)
        return current_rollout_asset.portfolio_result"""

        while True:
            # get the result of current asset
            # policy: maximize sharpe ratio with the return >= 4, return over drawdown >= 1, all weight <= 0.9
            result = current_rollout_asset.portfolio_result

            if result is not None:
                return result

            # if the result is None (does not follow the policy)
            # get all possible asset from the untried asset
            possible_asset = current_rollout_asset.get_possible_asset()  # /? rollout change get_possible

            if len(possible_asset) == 0:
                return 'fail'
            # randomly select one from all possible assets
            asset = self.rollout_policy(possible_asset)
            # add the selected asset into the asset list
            current_rollout_asset = current_rollout_asset.move(asset)

    def backpropagate(self, result):
        """
        backpropagate the result from current node up to the root
        :param result: 'conservative', 'aggressive', 'fail', None
        :return: None
        """
        # add the number of visit of current node
        self._number_of_visits += 1.
        print_flag = False

        # test if the result is None
        if result is not None and not result == 'fail':
            if self._portfolios_info is None:
                self._portfolios_info = pd.DataFrame(result)

                # test if the result is in the _results dictionary
                if result['Level'] not in self._results.keys():
                    # print("Result None Init TESTING!!!")
                    self._results[result['Level']] = 1.
                else:
                    # print("Result None Add TESTING!!!")
                    self._results[result['Level']] += 1.

            else:
                last_sharpe = self._portfolios_info['Sharpe'].mean()
                last_returns = self._portfolios_info['Returns'].mean()
                last_return_over_drawdown = self._portfolios_info['Return over drawdown'].mean()

                current_sharpe = result['Sharpe']
                current_returns = result['Returns']
                current_return_over_drawdown = result['Return over drawdown']

                if current_returns > last_returns:
                    # test if the result is in the _results dictionary
                    if result['Level'] not in self._results.keys():
                        # print("Result not None Init TESTING!!!")
                        self._results[result['Level']] = 1.
                    else:
                        # print("Result not None Add TESTING!!!")
                        self._results[result['Level']] += 1.

                    self._portfolios_info = pd.concat([self._portfolios_info, pd.DataFrame(result)])
                    print_flag = True
                    # print(self._portfolios_info)

        # test if current node has parent
        if self.parent:
            # backpropagate the result to the parent
            self.parent.backpropagate(result)

        # the current node does not has parent and the result is update
        elif print_flag:
            print(self._portfolios_info)
            self._portfolios_info.to_csv("final_" + risk_profile + "_portfolio.csv")
            print_flag = False
