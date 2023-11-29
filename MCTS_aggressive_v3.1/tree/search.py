import time


class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action
        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds
        Returns
        -------
        """

        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()  # select node to run
                reward = v.rollout()  # use the node to rollout
                v.backpropagate(reward)  # backpropagate the result to the parent
        else:
            for _ in range(0, simulations_number):
                # select a node to run rollout: terminal node, not fully expanded node
                v = self._tree_policy()
                # use the node to rollout and return a dictionary
                reward = v.rollout()
                # backpropagate the result to the parent
                v.backpropagate(reward)

        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.)
        return self.root.best_child()  # return the best child from root

    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        # call asset_list is satisfied, then call portfolio result to
        # get the result of the portfolio, if result is None, then expand this node or go deeper
        # if result is 'fail' or 'conservative' or 'portfolio', then stop iteration
        while not current_node.is_terminal_node():
            # test if the node is fully expanded by getting untried actions
            # (if None, get the actions, if no more action, return true)
            if not current_node.is_fully_expanded():
                # expand the node, select one asset from the untried asset,
                # add one asset to next node and add next node into this node's child node list
                # return next node
                return current_node.expand()
            else:
                # find the best child by Upper Confidence Bound
                current_node = current_node.best_child()
        return current_node
