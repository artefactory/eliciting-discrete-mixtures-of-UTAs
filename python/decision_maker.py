"""Implementation of Decision Maker with UTA decision function."""

import matplotlib.pyplot as plt
import numpy as np


class DecisionMaker:
    """Class representing a decision maker with a UTA decision function."""

    def __init__(self, n_criteria, n_pieces):
        """Initialize the decision maker with random slopes for the UTA function.

        Parameters
        ----------
        n_criteria : int
            Number of criteria for the decision maker.
        n_pieces : int
            Number of pieces for each criterion.
        """
        self.n_criteria = n_criteria
        self.n_pieces = n_pieces

        self.slopes = self.build_random_decision_function()
        self.breakpoints_x = np.linspace(0, self.n_pieces, self.n_pieces + 1)
        self.break_point_y = np.stack(
            [
                [0] + [np.sum(self.slopes[i][: j + 1]) for j in range(self.n_pieces)]
                for i in range(self.n_criteria)
            ]
        )
        self.total_n_answers = 0

    def build_random_decision_function(self):
        """Build a random decision function by generating random slopes for each criterion.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_criteria, n_pieces) containing the slopes for each criterion and piece.
        """
        slopes = []
        for i in range(self.n_criteria):
            crit_slopes = []
            for j in range(self.n_pieces):
                if i == 0 and j == 0:
                    crit_slopes.append(1.0)
                else:
                    crit_slopes.append(np.random.uniform(0, 10))
            slopes.append(crit_slopes)
        return np.stack(slopes)

    def plot_decision_function(self):
        """Plot the decision function for each criterion.

        The x-axis represents the value of the criterion, and the y-axis represents the utility.
        """
        x = np.linspace(0, self.n_pieces, self.n_pieces + 1)
        for i in range(self.n_criteria):
            y = [0]
            for j in range(self.n_pieces):
                y.append(y[-1] + self.slopes[i][j])
            plt.plot(x, y, label=f"Criterion {i+1}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Utility")
        plt.show()

    def get_ui(self, criterion, value):
        """Calculate the marginal utility of a given value for a specific criterion.

        Parameters
        ----------
        criterion : int
            The index of the criterion for which to calculate the utility.
        value : float
            The value for which to calculate the marginal utility.


        Returns
        -------
        float
            The marginal utility of the given value for the specified criterion.
        """
        marginal_utility = 0
        for i in range(self.n_pieces):
            sub_value = (
                np.max([np.min([1.0, value - i]), 0.0]) * self.slopes[criterion][i]
            )
            marginal_utility += sub_value
        return marginal_utility

    def answer(self, criterion_i, criterion_j, q_i, p_i, q_j):
        """Calculate the answer for a pairwise comparison between two criteria.

        Return x such that (q_i, p_i) is indifferent to (q_j, x) for the decision maker.
        If no such x exists, return None.

        Parameters
        ----------
        criterion_i : int
            The index of the first criterion.
        criterion_j : int
            The index of the second criterion.
        q_i : float
            The value of the first criterion for the first alternative.
        p_i : float
            The value of the first criterion for the second alternative.
        q_j : float
            The value of the second criterion for the first alternative.

        Returns
        -------
        float or None
            The value of the second criterion for the second alternative that makes the decision maker indifferent between the two alternatives, or None if no such value exists.
        """
        du = self.get_ui(criterion_i, p_i) - self.get_ui(criterion_i, q_i)
        u_j = self.get_ui(criterion_j, q_j)

        self.total_n_answers += 1

        if du > 0:
            if du > u_j:
                return None
            else:
                for i in range(self.n_pieces):
                    if (
                        u_j - self.break_point_y[criterion_j][i] >= du
                        and u_j - self.break_point_y[criterion_j][i + 1] < du
                    ):

                        dv = du - u_j + self.break_point_y[criterion_j][i + 1]
                        return i + 1 - (dv / self.slopes[criterion_j][i])
        else:
            if -du > (self.break_point_y[criterion_j][-1] - u_j):
                return None
            else:
                for i in range(self.n_pieces):
                    if (
                        self.break_point_y[criterion_j][i + 1] - u_j > -du
                        and self.break_point_y[criterion_j][i] - u_j <= -du
                    ):

                        dv = -du + u_j - self.break_point_y[criterion_j][i]
                        return i + dv / self.slopes[criterion_j][i]

    def get_total_n_answers(self):
        """Return the total number of answers given by the decision maker.

        Returns
        -------
        int
            The total number of answers given by the decision maker.
        """
        return self.total_n_answers


class HiddenDecisionMakers:
    """Class representing a collection of hidden decision makers.

    Mainly use to shuffle and hide the original decision makers when answering queries.
    """

    def __init__(self, n_dms, n_criteria, n_pieces):
        """Initialize Decision Makers.

        Parameters
        ----------
        n_dms : int
            Number of decision makers.
        n_criteria : int
            Number of criteria for each decision maker.
        n_pieces : int
            Number of pieces for each criterion.
        """
        self.dms = [DecisionMaker(n_criteria, n_pieces) for i in range(n_dms)]

    def answer(self, criterion_i, criterion_j, q_i, p_i, q_j):
        """Answer a query by shuffling the decision makers and returning their answers.

        Parameters
        ----------
        criterion_i : _type_
            _description_
        criterion_j : _type_
            _description_
        q_i : _type_
            _description_
        p_i : _type_
            _description_
        q_j : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        answers = [
            dm.answer(criterion_i, criterion_j, q_i, p_i, q_j) for dm in self.dms
        ]
        return np.random.permutation(answers)

    def get_total_n_answers(self):
        """Return the total number of answers given by all decision makers.

        Returns
        -------
        _type_
            _description_
        """
        return [dm.get_total_n_answers() for dm in self.dms]
