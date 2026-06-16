import numpy as np

from .decision_maker import DecisionMaker
from .query import SingleRectangleQuery, NeighboringRectanglesQuery


class UTAsIdentification:
    """Class for identifying the hidden decision makers' value functions using queries."""

    def __init__(self, n_criteria, n_pieces):
        """Initialize the identification process.

        Parameters
        ----------
        n_criteria : int
            Number of criteria for each decision maker.
        n_pieces : int
            Number of pieces for each criterion.
        """
        self.n_criteria = n_criteria
        self.n_pieces = n_pieces
        self.slopes = {
            "alpha": [
                [] for _ in range(n_criteria)
            ],
            "beta": [
                [] for _ in range(n_criteria)
            ]
        }
        self.slopes["alpha"][0].append(1.)
        self.slopes["beta"][0].append(1.)

    def identify_21(self, hdms):
        """Identify the slopes for the first two pieces of the first two criteria.

        Parameters
        ----------
        hdms : HiddenDecisionMakers
            The hidden decision makers to query.
        """
        query = SingleRectangleQuery(criterion_i=0, criterion_j=1, min_i=0, max_i=1/self.n_pieces, min_j=0, max_j=1/self.n_pieces)
        crits, I1, I2 = query.query_hidden_dms(hdms)

        self.slopes["alpha"][1].append((I1[0][0] - I1[1][0]) / (I1[1][1] - I1[0][1]))
        self.slopes["beta"][1].append((I2[0][0] - I2[1][0]) / (I2[1][1] - I2[0][1]))

    def identify_22(self, hdms):
        """Identify the slopes for the second piece of the first two criteria.

        Parameters
        ----------
        hdms : HiddenDecisionMakers
            The hidden decision makers to query.
        """
        a_step = 1 / self.n_pieces
        query_1 = SingleRectangleQuery(criterion_i=0, criterion_j=1, min_i=0, max_i=a_step, min_j=a_step, max_j=2*a_step)
        crits_1, I1_1, I2_1 = query_1.query_hidden_dms(hdms)

        unidentified_slope_1 = (I1_1[0][0] - I1_1[1][0]) / (I1_1[1][1] - I1_1[0][1])
        unidentified_slope_2 = (I2_1[0][0] - I2_1[1][0]) / (I2_1[1][1] - I2_1[0][1])


        query_2 = NeighboringRectanglesQuery(bridged_criterion=1, 
        constrained_criterion=0,
        bridged_breakpoint=a_step, 
        min_bridged=0, 
        max_bridged=2 *a_step, 
        min_squared=0, 
        max_squared=a_step)

        I1_2, I2_2 = query_2.query_hidden_dms(hdms)
        
        fixed_q = (I1_2[1][0] - I1_2[0][0]) / (I1_2[0][1] - a_step)
        u_quotient_21 = (I1_2[1][1] - a_step) / (I1_2[0][1] - a_step)
        u_quotient_22 = (I2_2[1][1] - a_step) / (I2_2[0][1] - a_step)

        fixed_q_1 = (I1_2[0][0] - I1_2[1][0]) / (I1_2[1][1] - a_step)
        fixed_q_2 = (I2_2[0][0] - I2_2[1][0]) / (I2_2[1][1] - a_step)
        u_quotient_21 = (I1_2[0][1] - a_step) / (I1_2[1][1] - a_step)
        u_quotient_22 = (I2_2[0][1] - a_step) / (I2_2[1][1] - a_step)

        possible_slopes_1 = [
            fixed_q_1 + u_quotient_21 * self.slopes["alpha"][1][0],
            fixed_q_2 + u_quotient_22 * self.slopes["beta"][1][0],
        ]
        possible_slopes_2 = [
            fixed_q_2 + u_quotient_22 * self.slopes["alpha"][1][0],
            fixed_q_1 + u_quotient_21 * self.slopes["beta"][1][0],
        ]

        case_1 = np.min([
            np.abs(possible_slopes_1[0] -unidentified_slope_1) + np.abs(possible_slopes_1[1] -unidentified_slope_2), 
            np.abs(possible_slopes_1[1] -unidentified_slope_1) + np.abs(possible_slopes_1[0] -unidentified_slope_2)
        ])

        case_2 = np.min([
            np.abs(possible_slopes_2[0] -unidentified_slope_1) + np.abs(possible_slopes_2[1] -unidentified_slope_2), 
            np.abs(possible_slopes_2[1] -unidentified_slope_1) + np.abs(possible_slopes_2[0] -unidentified_slope_2)
        ])
        if case_1 < case_2:
            self.slopes["alpha"][1].append(possible_slopes_1[0])
            self.slopes["beta"][1].append(possible_slopes_1[1])

        else:
            self.slopes["alpha"][1].append(possible_slopes_2[0])
            self.slopes["beta"][1].append(possible_slopes_2[1])

    def generic_identification(self, hdms, criterion, square):
        """Identify the slopes for a given criterion and square.

        Parameters
        ----------
        hdms : HiddenDecisionMakers
            The hidden decision makers to query.
        criterion : int
            The criterion for which to identify the slopes.
        square : int
            The square for which to identify the slopes.
        """
        a_step = 1 / self.n_pieces
        if criterion == 0:
            opposite_criterion = 1
        else:
            opposite_criterion = 0

        query = SingleRectangleQuery(criterion_i=criterion,
        criterion_j=opposite_criterion,
        min_i=square * a_step,
        max_i=(square+1) * a_step,
        min_j=0,
        max_j=a_step)
        crits, I1, I2 = query.query_hidden_dms(hdms)

        if opposite_criterion == 0:
            slopes_values = [(I1[1][1] - I1[0][1]) / (I1[0][0] - I1[1][0]), ((I2[1][1] - I2[0][1]) / (I2[0][0] - I2[1][0]))]
        else:
            slopes_values = [[((I1[1][1] - I1[0][1]) / (I1[0][0] - I1[1][0])) * self.slopes["alpha"][opposite_criterion][0], 
            ((I2[1][1] - I2[0][1]) / (I2[0][0] - I2[1][0])) * self.slopes["beta"][opposite_criterion][0]],
            [((I2[1][1] - I2[0][1]) / (I2[0][0] - I2[1][0])) * self.slopes["alpha"][opposite_criterion][0], 
            ((I1[1][1] - I1[0][1]) / (I1[0][0] - I1[1][0])) * self.slopes["beta"][opposite_criterion][0]],]
        
        query = SingleRectangleQuery(criterion_i=criterion,
        criterion_j=opposite_criterion,
        min_i=square * a_step,
        max_i=(square+1) * a_step,
        min_j=a_step,
        max_j=2 * a_step)
        crits, I1, I2 = query.query_hidden_dms(hdms)

        possible_slopes = [((I1[1][1] - I1[0][1]) / (I1[0][0] - I1[1][0])) * self.slopes["alpha"][opposite_criterion][1], 
        ((I2[1][1] - I2[0][1]) / (I2[0][0] - I2[1][0]))* self.slopes["beta"][opposite_criterion][1]]
        other_possible_slopes = [((I1[1][1] - I1[0][1]) / (I1[0][0] - I1[1][0])) * self.slopes["beta"][opposite_criterion][1], 
        ((I2[1][1] - I2[0][1]) / (I2[0][0] - I2[1][0]))* self.slopes["alpha"][opposite_criterion][1]]

        if opposite_criterion == 0:
            case_1 = np.min([
                np.abs(possible_slopes[0] - slopes_values[0]) + np.abs(possible_slopes[1] - slopes_values[1]),
                np.abs(possible_slopes[1] - slopes_values[0]) + np.abs(possible_slopes[0] - slopes_values[1])
            ])

            case_2 = np.min([
                np.abs(other_possible_slopes[1] - slopes_values[0]) + np.abs(other_possible_slopes[0] - slopes_values[1]),
                np.abs(other_possible_slopes[0] - slopes_values[0]) + np.abs(other_possible_slopes[1] - slopes_values[1])
            ])
        else:
            case_1 = np.min([
                np.abs(possible_slopes[0] - slopes_values[0][0]) + np.abs(possible_slopes[1] - slopes_values[0][1]),
                np.abs(possible_slopes[0] - slopes_values[1][0]) + np.abs(possible_slopes[1] - slopes_values[1][1])
            ])

            case_2 = np.min([
                np.abs(other_possible_slopes[1] - slopes_values[0][0]) + np.abs(other_possible_slopes[0] - slopes_values[0][1]),
                np.abs(other_possible_slopes[1] - slopes_values[1][0]) + np.abs(other_possible_slopes[0] - slopes_values[1][1])
            ])

        if np.min([case_1, case_2]) > 0.1:
            print("Warning: The identified slopes are not consistent with the previous ones. Please check the queries and the hidden decision makers.")
            print("Slopes values: ", slopes_values)
            print("Possible slopes: ", possible_slopes)
            print("Other possible slopes: ", other_possible_slopes)
            print("Case 1: ", case_1)
            print("Case 2: ", case_2)
        if case_1 < case_2:
            self.slopes["alpha"][criterion].append(possible_slopes[0])
            self.slopes["beta"][criterion].append(possible_slopes[1])
        else:
            self.slopes["alpha"][criterion].append(other_possible_slopes[1])
            self.slopes["beta"][criterion].append(other_possible_slopes[0])

    def identify_hdms(self, hdms):
        """Identify the hidden decision makers' value functions using queries.

        Parameters
        ----------
        hdms : HiddenDecisionMakers
            The hidden decision makers to query.
            
        Returns
        -------
        dict
            The identified coefficients for the hidden decision makers' value functions.
        """
        self.identify_21(hdms)
        self.identify_22(hdms)
        self.generic_identification(hdms=hdms, criterion=0, square=1)

        for i in range(2, self.n_pieces):
            self.generic_identification(hdms=hdms, criterion=0, square=i)
            self.generic_identification(hdms=hdms, criterion=1, square=i)
            
        for i in range(2, self.n_criteria):
            for j in range(self.n_pieces):
                self.generic_identification(hdms=hdms, criterion=i, square=j)
        
        # Normalization
        coefficients = {"alpha": [[0.] for _ in range(self.n_criteria)], "beta": [[0.] for _ in range(self.n_criteria)]}
        step = 1 / self.n_pieces

        for crit_j in range(self.n_criteria):
            for i in self.slopes["alpha"][crit_j]:
                coefficients["alpha"][crit_j].append(i * step + coefficients["alpha"][crit_j][-1])
            for i in self.slopes["beta"][crit_j]:
                coefficients["beta"][crit_j].append(i * step + coefficients["beta"][crit_j][-1])
        coefficients["alpha"] = np.array(coefficients["alpha"]).astype("float32")
        coefficients["beta"] = np.array(coefficients["beta"]).astype("float32")  
        coefficients["alpha"] = np.array(coefficients["alpha"]).astype("float32") / np.sum(coefficients["alpha"][:, -1])
        coefficients["beta"] = np.array(coefficients["beta"]).astype("float32") / np.sum(coefficients["beta"][:, -1])
        self.coefficients = coefficients
        return coefficients

