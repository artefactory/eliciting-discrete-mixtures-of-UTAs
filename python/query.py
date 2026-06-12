"""Implementation of the different types of queries."""

import numpy as np


class SquaredQuery:
    """Class representing a squared query between two criteria."""

    def __init__(self, criterion_i, criterion_j, min_i, max_i, min_j, max_j):
        """Initialize a squared query which will be constrained in [min_i, max_i] x [min_j, max_j].

        Parameters
        ----------
        criterion_i : int
            The index of the first criterion.
        criterion_j : int
            The index of the second criterion.
        min_i : float
            The minimum value for the first criterion.
        max_i : float
            The maximum value for the first criterion.
        min_j : float
            The minimum value for the second criterion.
        max_j : float
            The maximum value for the second criterion.
        """
        self.criterion_i = criterion_i
        self.criterion_j = criterion_j
        self.min_i = min_i
        self.max_i = max_i
        self.min_j = min_j
        self.max_j = max_j

    def query_hidden_dms(self, hdm):
        """Query the hidden decision makers with the squared query.
        
        Parameters
        ----------
        hdm : HiddenDecisionMakers
            The hidden decision makers to query.
        
        Returns
        -------
        crits : list of int
            The indices of the criteria involved in the query.
        I1 : list of list of float
            The answers of the hidden decision makers to the query for the first alternative.
        I2 : list of list of float
            The answers of the hidden decision makers to the query for the second alternative.
        """
        # First Query
        # (q_i, q_j) ~ (q_j, ?)
        q_i = self.max_i
        p_i = self.min_i
        q_j = self.min_j

        answer_1 = hdm.answer(
            criterion_i=self.criterion_i,
            criterion_j=self.criterion_j,
            q_i=q_i,
            q_j=q_j,
            p_i=p_i,
        )
        answer_1[answer_1 == None] = self.max_j + 1

        if np.sum(answer_1 > self.max_j) == 2:
            answer_1 = hdm.answer(
                criterion_i=self.criterion_j,
                criterion_j=self.criterion_i,
                q_i=self.max_j,
                q_j=self.min_i,
                p_i=self.min_j,
            )
            assert None not in answer_1, answer_1
            return (
                [self.criterion_i, self.criterion_j],
                [
                    [self.min_i, self.max_j],
                    [answer_1[0], self.min_j],
                ],
                [[self.min_i, self.max_j], [answer_1[1], self.min_j]],
            )

        elif np.sum(answer_1 > self.max_j) == 1:
            q_i = np.floor(np.min(answer_1) * 100) / 100
            p_i = self.min_j
            q_j = self.min_i
            answer_1 = hdm.answer(
                criterion_i=self.criterion_j,
                criterion_j=self.criterion_i,
                q_i=q_i,
                q_j=q_j,
                p_i=p_i,
            )
            assert None not in answer_1, answer_1
            return (
                [self.criterion_i, self.criterion_j],
                [
                    [self.min_i, q_i],
                    [answer_1[0], self.min_j],
                ],
                [[self.min_i, q_i], [answer_1[1], self.min_j]],
            )

        else:
            return (
                [self.criterion_i, self.criterion_j],
                [[self.max_i, self.min_j], [self.min_i, answer_1[0]]],
                [[self.max_i, self.min_j], [self.min_i, answer_1[1]]],
            )


class NeightboringRectanglesQuery:
    """Class representing a query between two neighboring rectangles."""
    def __init__(
        self,
        bridged_criterion,
        constrained_criterion,
        bridged_breakpoint,
        min_bridged,
        max_bridged,
        min_squared,
        max_squared,
    ):
        """Initialize a query between two neighboring rectangles.

        Parameters
        ----------
        bridged_criterion : int
            The index of the bridged criterion.
        constrained_criterion : int
            The index of the constrained criterion.
        bridged_breakpoint : float
            The value of the bridged breakpoint.
        min_bridged : float
            The minimum value of the bridged criterion.
        max_bridged : float
            The maximum value of the bridged criterion.
        min_squared : float
            The minimum value of the squared criterion.
        max_squared : float
            The maximum value of the squared criterion.
        """
        self.bridged_criterion = bridged_criterion
        self.constrained_criterion = constrained_criterion

        # maybe do better
        self.bridged_breakpoint = bridged_breakpoint
        self.min_bridged = min_bridged
        self.max_bridged = max_bridged
        self.min_squared = min_squared
        self.max_squared = max_squared

    def query_hidden_dms(self, hdm):
        """Query the hidden decision makers with the neighboring rectangles query.

        Parameters
        ----------
        hdm : HiddenDecisionMakers
            The hidden decision makers to query.
        
        Returns
        -------
        I1 : list of list of float
            The answers of the hidden decision makers to the query for the first alternative.
        I2 : list of list of float
            The answers of the hidden decision makers to the query for the second alternative.
        """
        # First Query
        # (q_i, q_j) ~ (q_j, ?)
        q_i = self.min_squared + (self.max_squared - self.min_squared) / 2
        p_i = self.min_squared
        q_j = (self.bridged_breakpoint - self.min_bridged) / 2 + self.min_bridged

        answer_1 = hdm.answer(
            criterion_i=self.constrained_criterion,
            criterion_j=self.bridged_criterion,
            q_i=q_i,
            q_j=q_j,
            p_i=p_i,
        )

        while True:
            if (
                np.sum(answer_1 > self.bridged_breakpoint) == 2
                and np.sum(answer_1 <= self.max_bridged) == 2
            ):
                return ([q_i, q_j], [p_i, answer_1[0]]), (
                    [q_i, q_j],
                    [p_i, answer_1[1]],
                )

            elif np.sum(answer_1 > self.bridged_breakpoint) < 2:
                q_j = (
                    q_j
                    + (self.bridged_breakpoint - np.min(answer_1))
                    + (np.min(answer_1) - q_j) / 2
                )
                answer_1 = hdm.answer(
                    criterion_i=self.constrained_criterion,
                    criterion_j=self.bridged_criterion,
                    q_i=q_i,
                    q_j=q_j,
                    p_i=p_i,
                )

            else:
                q_i = q_i - (q_i - self.min_squared) / 2
                answer_1 = hdm.answer(
                    criterion_i=self.constrained_criterion,
                    criterion_j=self.bridged_criterion,
                    q_i=q_i,
                    q_j=q_j,
                    p_i=p_i,
                )
