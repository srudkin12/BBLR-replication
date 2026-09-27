import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression


# Define all DGP functions
def generate_data(n, dgp_name, d, noise_var = 0.2, seed=42, x_distribution='normal'):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # TODO: other distribution: lognormal, uniform etc.
    match x_distribution:
        case 'normal':
            x = np.random.normal(0, 1, (n, d))
        case 'uniform':
            x = np.random.uniform(0, 1, (n, d))
        case 'lognormal':
            x = np.random.lognormal(0, 1, (n, d))
        case 'weibull':
            x = np.random.weibull(0.7, (n, d))
        case 'pareto':
            x = np.random.pareto(0.7, (n, d))
        case _:
            raise ValueError(f"Unknown distribution: {x_distribution}")
    eps = np.random.normal(0, noise_var, n)

    match dgp_name:
        case 'UA':
            y = x[:,0] + eps
        case 'UB':
            y = x[:,0]**2 + eps
        case 'UC':
            # Step function with 7 steps: nonincreasing on (-∞,0), nondecreasing on (0,∞)
            # Define step boundaries and values
            steps = np.array([-1.5, -1, -0.5, 0.5, 1, 1.5])  # 6 boundaries create 7 steps
            step_values = np.array([1.5, 1, 0.5, 0, 0.5, 1, 1.5])  # 7 step values

            # Use np.digitize to assign each x value to a step
            step_indices = np.digitize(x[:,0], steps)
            y = step_values[step_indices] + eps

        case 'UD':
            # Linear relation with one discontinuity at x=0
            # Different coefficients before and after discontinuity
            coeff_neg = -2.0  # coefficient for x < 0
            coeff_pos = 1.5   # coefficient for x >= 0
            y = np.where(x[:,0] < 0, coeff_neg * x[:,0], coeff_pos * x[:,0]) + 2 * (x[:,0] > 0) + eps

        case 'UE':
            # Linear relation with two discontinuities
            # Three different coefficients: positive, negative, positive
            breakpoint1 = -1.0
            breakpoint2 = 1.0
            coeff1 = 2.0   # positive coefficient for x < -1
            coeff2 = -1.5  # negative coefficient for -1 <= x < 1
            coeff3 = 1.8   # positive coefficient for x >= 1

            y = (np.where(x[:,0] < breakpoint1,
                         coeff1 * (x[:,0] - breakpoint1),
                         np.where(x[:,0] < breakpoint2,
                                  coeff2 * (x[:,0] - breakpoint1),
                                  coeff3 * (x[:,0] - breakpoint2) + coeff2 * (breakpoint2 - breakpoint1)))
                 + 2 * (x[:,0] > -1) + (x[:,0] > 1) + eps)

        case 'UF':
            # Sine wave with three hills
            # This creates approximately 3 peaks in the range where most standard normal values lie
            frequency = 1  # controls number of hills
            amplitude = 2.0  # controls height
            y = amplitude * np.sin(frequency * np.pi * x[:,0]) + eps
        case 'UG':
            y = np.exp(x[:,0]) + eps
        case 'UH':
            y = np.sin(x[:,0]) + x[:,0] + eps  # case 1 in Fig. 3 in TabPFN paper
        case 'UI':
            y = x[:,0] ** 3 + eps
        case 'UJ':
            # x are 500 equally distance points in the range [0,2]
            y = np.arange(0, 2, 2/n) ** (1/2) + eps 
        case 'UK':
            M_limits = 1
            m_limits = 0.2
            a0 = np.random.uniform(-M_limits, M_limits)
            a1 = np.random.uniform(-M_limits, M_limits)
            a2 = np.random.uniform(-m_limits, m_limits)
            # a3 = np.random.uniform(-m_limits, m_limits)
            a3 = 0
            a_vector = np.array([a0, a1, a2, a3])
            y = a0 + a1 * x[:,0] + a2 * x[:,0]**2 + a3 * x[:,0]**3 + eps
        case 'UL':
            m_limits = 0.2
            a = np.random.uniform(-m_limits, m_limits)  # controls bend; 0 = purely linear
            a_vector = np.array([a])
            y = x[:,0] + a * x[:,0]**2 + eps
        case 'UM':
            # Partially linear relation with one kink at x=0: decreasing then increasing
            # Random slopes control the (varying) angle between the two line segments
            slope_min, slope_max = 0.3, 2.0
            slope_neg = -np.random.uniform(slope_min, slope_max)  # slope for x < 0
            slope_pos = np.random.uniform(slope_min, slope_max)   # slope for x >= 0
            a_vector = np.array([slope_neg, slope_pos])
            y = np.where(x[:,0] < 0, slope_neg * x[:,0], slope_pos * x[:,0]) + eps
        case 'UN':
            # Constant-slope line with a random-height jump discontinuity at x=1
            jump_max = 2.0
            jump_height = np.random.uniform(0, jump_max)
            a_vector = np.array([jump_height])
            y = x[:,0] + jump_height * (x[:,0] >= 1) + eps
        case 'UO':
            # Combination of UM and UN: kinked slope at x=1 (decreasing then increasing)
            # plus a random-height jump discontinuity at x=1
            slope_min, slope_max = 0.3, 2.0
            jump_max = 2.0
            breakpoint = 1.0
            slope_before = -np.random.uniform(slope_min, slope_max)  # slope for x < 1
            slope_after = np.random.uniform(slope_min, slope_max)    # slope for x >= 1
            jump_height = np.random.uniform(0, jump_max)
            a_vector = np.array([slope_before, slope_after, jump_height])
            y = (np.where(x[:,0] < breakpoint,
                          slope_before * (x[:,0] - breakpoint),
                          slope_after * (x[:,0] - breakpoint))
                 + jump_height * (x[:,0] >= breakpoint) + eps)
        case 'RA':
            # Multidimensional analogue of UL: quadratic bend added independently
            # to X1, X2, X3 (random coefficients), X4 irrelevant
            m_limits = 0.2
            a1 = np.random.uniform(-m_limits, m_limits)
            a2 = np.random.uniform(-m_limits, m_limits)
            a3 = np.random.uniform(-m_limits, m_limits)
            a_vector = np.array([a1, a2, a3])
            y = (x[:,0] + x[:,1] + x[:,2]
                 + a1 * x[:,0]**2 + a2 * x[:,1]**2 + a3 * x[:,2]**2 + eps)
        case 'RB':
            # Multidimensional analogue of UM: kink at 0 with random slopes,
            # applied independently to X1, X2, X3, X4 irrelevant
            slope_min, slope_max = 0.3, 2.0
            slope_neg1 = -np.random.uniform(slope_min, slope_max)
            slope_pos1 = np.random.uniform(slope_min, slope_max)
            slope_neg2 = -np.random.uniform(slope_min, slope_max)
            slope_pos2 = np.random.uniform(slope_min, slope_max)
            slope_neg3 = -np.random.uniform(slope_min, slope_max)
            slope_pos3 = np.random.uniform(slope_min, slope_max)
            a_vector = np.array([slope_neg1, slope_pos1, slope_neg2, slope_pos2, slope_neg3, slope_pos3])
            y = (np.where(x[:,0] < 0, slope_neg1 * x[:,0], slope_pos1 * x[:,0])
                 + np.where(x[:,1] < 0, slope_neg2 * x[:,1], slope_pos2 * x[:,1])
                 + np.where(x[:,2] < 0, slope_neg3 * x[:,2], slope_pos3 * x[:,2])
                 + eps)
        case 'RC':
            # Multidimensional analogue of UN: random-height jump discontinuity
            # at x=1 applied independently to X1, X2, X3, X4 irrelevant
            jump_max = 2.0
            jump_height1 = np.random.uniform(0, jump_max)
            jump_height2 = np.random.uniform(0, jump_max)
            jump_height3 = np.random.uniform(0, jump_max)
            a_vector = np.array([jump_height1, jump_height2, jump_height3])
            y = (x[:,0] + x[:,1] + x[:,2]
                 + jump_height1 * (x[:,0] >= 1)
                 + jump_height2 * (x[:,1] >= 1)
                 + jump_height3 * (x[:,2] >= 1)
                 + eps)
        case 'RD':
            # Multidimensional analogue of UO: kinked slope plus a random-height
            # jump at x=1, applied independently to X1, X2, X3, X4 irrelevant
            slope_min, slope_max = 0.3, 2.0
            jump_max = 2.0
            breakpoint = 1.0
            slope_before1 = -np.random.uniform(slope_min, slope_max)
            slope_after1 = np.random.uniform(slope_min, slope_max)
            slope_before2 = -np.random.uniform(slope_min, slope_max)
            slope_after2 = np.random.uniform(slope_min, slope_max)
            slope_before3 = -np.random.uniform(slope_min, slope_max)
            slope_after3 = np.random.uniform(slope_min, slope_max)
            jump_height1 = np.random.uniform(0, jump_max)
            jump_height2 = np.random.uniform(0, jump_max)
            jump_height3 = np.random.uniform(0, jump_max)
            a_vector = np.array([slope_before1, slope_after1, slope_before2, slope_after2,
                                  slope_before3, slope_after3, jump_height1, jump_height2, jump_height3])
            y = (np.where(x[:,0] < breakpoint, slope_before1 * (x[:,0] - breakpoint), slope_after1 * (x[:,0] - breakpoint))
                 + jump_height1 * (x[:,0] >= breakpoint)
                 + np.where(x[:,1] < breakpoint, slope_before2 * (x[:,1] - breakpoint), slope_after2 * (x[:,1] - breakpoint))
                 + jump_height2 * (x[:,1] >= breakpoint)
                 + np.where(x[:,2] < breakpoint, slope_before3 * (x[:,2] - breakpoint), slope_after3 * (x[:,2] - breakpoint))
                 + jump_height3 * (x[:,2] >= breakpoint)
                 + eps)
        case 'BA':
            y = x[:,0] + x[:,1] + eps
        case 'BB':
            y = x[:,0]**2 + x[:,1]**2 + eps
        case 'BC':
            y = x[:,0] + x[:,1] + x[:,0] * x[:,1] * eps
        case 'BD':
            y = x[:,0] + x[:,1] + 2 * x[:,0] * x[:,1] * eps
        case 'BE':
            y = x[:,0] + x[:,1] + 0.5 * x[:,0] * x[:,1] * eps
        case 'BF':
            y = x[:,0]**2 + x[:,1] + eps
        case 'BG':
            y = x[:,0]**2 + np.sin(x[:,1]) + eps
        case 'BH':
            y = x[:,0]**2 + np.log(np.abs(x[:,1])) + eps
        case 'BI':
            y = np.abs(x[:,0]) + np.exp(x[:,1]) + eps
        case 'BJ':
            y = x[:,0]**2 + x[:,1] + x[:,0] * x[:,1] + eps
        case 'BK':
            y = x[:,0]**2 + x[:,1] + eps
        case 'CA':
            y = x[:,0] + x[:,1]**2 + np.exp(x[:,2]) + eps  # renamed in the paper
        case 'MA':
            y = np.exp(0.2*x[:,0]) + 2*x[:,1] - x[:,2]**2 + eps  # + 0 * x[:,3]
        case 'MB':
            y = np.sqrt(np.abs(x[:,0]))  + 2*x[:,1] + np.log(np.maximum(7 + x[:,2], 0.0001)) + eps  # + 0 * x[:,3]
        case 'MC':
            # Pure interaction + quadratic; X4 irrelevant
            y = x[:,0] * x[:,1] + x[:,2]**2 + eps  # + 0 * x[:,3]
        case 'MD':
            # Trigonometric nonlinearities + linear interaction
            y = np.sin(x[:,0]) + np.cos(x[:,1]) + x[:,2] * x[:,3] + eps
        case 'ME':
            # Your requested mixed example: linear + bilinear + quadratic + trig interaction
            y = x[:,0] + x[:,0]*x[:,1] + x[:,2]**2 + np.sin(x[:,2]*x[:,3]) + eps
        case 'MF':
            # Log-transform + cubic + bilinear interaction; X4 irrelevant
            y = np.log1p(np.abs(x[:,0])) + x[:,1]**3 + x[:,1]*x[:,2] + eps  # + 0 * x[:,3]
        case 'MG':
            # Rational + Gaussian bump + linear; complex curvature
            y = x[:,0] / (1 + x[:,1]**2) + np.exp(-x[:,2]**2) + x[:,3] + eps
        case 'MH':
            # Cross-term + absolute difference + quadratic; all vars active
            y = x[:,0]*x[:,2] + np.abs(x[:,1] - x[:,3]) + x[:,2]**2 + eps
        case 'MI':
            # Tanh of sum + product with trig; X4 irrelevant
            y = np.tanh(x[:,0] + x[:,1]) + x[:,2]*np.sin(x[:,0]) + eps  # + 0 * x[:,3]
        case 'MJ':
            # Cubic interaction + root + cubic; all vars active
            y = x[:,0]**2 * x[:,1] + np.sqrt(np.abs(x[:,2])) + x[:,3]**3 + eps
        case 'MK':
            # Exponential of product (mild interaction) + log regularisation
            y = np.exp(x[:,0]*x[:,1]) + x[:,2] + np.log1p(x[:,3]**2) + eps
        case 'ML':
            # Three-way trig interactions; X4 irrelevant
            y = x[:,0]*np.sin(x[:,1]) + x[:,1]*np.cos(x[:,2]) + x[:,0]*x[:,2] + eps  # + 0 * x[:,3]
        case 'MM':
            # Log-transform + cubic + bilinear interaction; X4 irrelevant
            y = np.log1p(np.abs(x[:,0])) + 0.2 * x[:,1]**3 + x[:,1]*x[:,2] + eps  # + 0 * x[:,3]
        case 'MN':
            # Cubic interaction + root + cubic; all vars active
            y = x[:,0]**2 * x[:,1] + np.sqrt(np.abs(x[:,2])) + 0.3 * x[:,3]**3 + eps
        case 'MO':
            y = np.exp(0.2*x[:,0]) + 2*x[:,1] - 0.25 * x[:,2]**2 + eps  # + 0 * x[:,3]
        case 'MP':
            y = 5 + np.exp(0.2*x[:,0]) + 2*x[:,1] - x[:,2]**2 + eps  # + 0 * x[:,3]
        case 'MQ':
            bp1 = -1.0
            bp2 = 1.0

            beta_r1 = np.array([ 2.0,  1.5, -1.0,  0.5])
            beta_r2 = np.array([-1.5, -1.0,  2.0, -0.8])
            beta_r3 = np.array([ 1.8,  0.5, -1.5,  1.2])

            r1 = x[:, 0] < bp1
            r2 = (x[:, 0] >= bp1) & (x[:, 0] < bp2)
            r3 = x[:, 0] >= bp2

            y = (np.where(r1,
                          beta_r1[0] * (x[:, 0] - bp1),
                 np.where(r2,
                          beta_r2[0] * (x[:, 0] - bp1),
                          beta_r3[0] * (x[:, 0] - bp2) + beta_r2[0] * (bp2 - bp1)))
                 + (r1 * beta_r1[1] + r2 * beta_r2[1] + r3 * beta_r3[1]) * x[:, 1]
                 + (r1 * beta_r1[2] + r2 * beta_r2[2] + r3 * beta_r3[2]) * x[:, 2]
                 + (r1 * beta_r1[3] + r2 * beta_r2[3] + r3 * beta_r3[3]) * x[:, 3]
                 + 2 * (x[:, 0] > bp1) + 1 * (x[:, 0] > bp2) + eps)
        case 'MR':
            # https://www.desmos.com/calculator/1pnyjeyahw
            x = np.random.uniform(0, 2, (n, d))
            y = 2*np.log1p(x[:,0]+1) + np.sin(0.7*x[:,1]) + 0.3*(x[:,2]-1)**3 + 0.3*(x[:,2]-1) + x[:,3] + eps
        case 'MS':
            y = x[:,0] + 2*x[:,1] + 0.5*x[:,2] + x[:,3]**2 + eps
        case 'MT':
            y = x[:,0] + 2*x[:,1] + 0.5*x[:,2] + x[:,3]**2 + eps
        case 'Z1':
            # y = 0.4*X1^2 + eps
            y = 0.4 * x[:,0]**2 + eps
        case 'Z2':
            # y = 0.4*X1^2 + 0.4*X2^2 + eps
            y = 0.4 * x[:,0]**2 + 0.4 * x[:,1]**2 + eps
        case 'Z3':
            # y = 0.4*X1^2 + ... + 0.4*X3^2 + eps
            y = 0.4 * x[:,0]**2 + 0.4 * x[:,1]**2 + 0.4 * x[:,2]**2 + eps
        case 'Z4':
            # y = 0.4*X1^2 + ... + 0.4*X4^2 + eps
            y = 0.4 * x[:,0]**2 + 0.4 * x[:,1]**2 + 0.4 * x[:,2]**2 + 0.4 * x[:,3]**2 + eps
        case 'Z5':
            # y = 0.4*X1^2 + ... + 0.4*X5^2 + eps
            y = 0.4 * x[:,0]**2 + 0.4 * x[:,1]**2 + 0.4 * x[:,2]**2 + 0.4 * x[:,3]**2 + 0.4 * x[:,4]**2 + eps
        case 'friedman1':
            x, y = make_friedman1(n_samples=n, n_features=d, noise=noise_var, random_state=seed)
        case _:
            raise ValueError(f"Unknown DGP: {dgp_name}")

    data_dict = {f'X{i+1}': x[:,i] for i in range(d)}
    data_dict['y'] = y

    if dgp_name == 'BK':
        data_dict['Z1'] = x[:,0]**2
        data_dict['Z2'] = x[:,1]
    if dgp_name == 'MT':
        data_dict['Z1'] = x[:,0]
        data_dict['Z2'] = x[:,1]
        data_dict['Z3'] = x[:,2]
        data_dict['Z4'] = x[:,3]**2

    if dgp_name in ('UK', 'UL', 'UM', 'UN', 'UO', 'RA', 'RB', 'RC', 'RD'):
        return pd.DataFrame(data_dict), a_vector
    else:
        return pd.DataFrame(data_dict)



def compute_true_coefs(X, result_df, dgp_name, a_vector=None, **kwargs):
    X_size = len(X)
    match dgp_name:
        case 'UA':
            B_true = {'B1_true': np.ones(X_size)}
        case 'UB':
            B_true = {'B1_true': 2 * result_df['X1']}
        case 'UC':
            B_true = {'B1_true': np.zeros(X_size)}
        case 'UD':
            coeff_neg = -2.0
            coeff_pos = 1.5
            B_true = {'B1_true': np.where(result_df['X1'] < 0, coeff_neg, coeff_pos)}
        case 'UE':
            breakpoint1 = -1.0
            breakpoint2 = 1.0
            coeff1 = 2.0
            coeff2 = -1.5
            coeff3 = 1.8
            B_true = {'B1_true': np.where(result_df['X1'] < breakpoint1,
                                          coeff1,
                                          np.where(result_df['X1'] < breakpoint2,
                                                   coeff2,
                                                   coeff3))}
        case 'UF':
            frequency = 1
            amplitude = 2.0
            B_true = {'B1_true': amplitude * frequency * np.pi * np.cos(frequency * np.pi * result_df['X1'])}
        case 'UG':
            B_true = {'B1_true': np.exp(result_df['X1'])}
        case 'UH':
            B_true = {'B1_true': np.cos(result_df['X1']) + 1}
        case 'UI':
            B_true = {'B1_true': 3 * result_df['X1'] ** 2}
        case 'UJ':
            B_true = {'B1_true': 0.5 * result_df['X1'] ** (-1/2)}
        case 'UK':
            a0 = a_vector[0]
            a1 = a_vector[1]
            a2 = a_vector[2]
            a3 = a_vector[3]
            B_true = {'B1_true': a1 + 2*a2*result_df['X1'] + 3*a3*result_df['X1']**2}
        case 'UL':
            a = a_vector[0]
            B_true = {'B1_true': 1 + 2 * a * result_df['X1']}
        case 'UM':
            slope_neg = a_vector[0]
            slope_pos = a_vector[1]
            B_true = {'B1_true': np.where(result_df['X1'] < 0, slope_neg, slope_pos)}
        case 'UN':
            B_true = {'B1_true': np.ones(X_size)}
        case 'UO':
            slope_before = a_vector[0]
            slope_after = a_vector[1]
            B_true = {'B1_true': np.where(result_df['X1'] < 1.0, slope_before, slope_after)}
        case 'RA':
            a1, a2, a3 = a_vector[0], a_vector[1], a_vector[2]
            B_true = {
                'B1_true': 1 + 2 * a1 * result_df['X1'],
                'B2_true': 1 + 2 * a2 * result_df['X2'],
                'B3_true': 1 + 2 * a3 * result_df['X3'],
                'B4_true': np.zeros(X_size)
            }
        case 'RB':
            slope_neg1, slope_pos1, slope_neg2, slope_pos2, slope_neg3, slope_pos3 = a_vector
            B_true = {
                'B1_true': np.where(result_df['X1'] < 0, slope_neg1, slope_pos1),
                'B2_true': np.where(result_df['X2'] < 0, slope_neg2, slope_pos2),
                'B3_true': np.where(result_df['X3'] < 0, slope_neg3, slope_pos3),
                'B4_true': np.zeros(X_size)
            }
        case 'RC':
            B_true = {
                'B1_true': np.ones(X_size),
                'B2_true': np.ones(X_size),
                'B3_true': np.ones(X_size),
                'B4_true': np.zeros(X_size)
            }
        case 'RD':
            (slope_before1, slope_after1, slope_before2, slope_after2,
             slope_before3, slope_after3, jump_height1, jump_height2, jump_height3) = a_vector
            B_true = {
                'B1_true': np.where(result_df['X1'] < 1.0, slope_before1, slope_after1),
                'B2_true': np.where(result_df['X2'] < 1.0, slope_before2, slope_after2),
                'B3_true': np.where(result_df['X3'] < 1.0, slope_before3, slope_after3),
                'B4_true': np.zeros(X_size)
            }
        case 'BA':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': np.ones(X_size)}
        case 'BB':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': 2 * result_df['X2']}
        case 'BC':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': np.ones(X_size)}
        case 'BD':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': np.ones(X_size)}
        case 'BE':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': np.ones(X_size)}
        case 'BF':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': np.ones(X_size)}
        case 'BG':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': np.cos(result_df['X2'])}
        case 'BH':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': 1 / result_df['X2']}
        case 'BI':
            B_true = {'B1_true': np.sign(result_df['X1']), 'B2_true': np.exp(result_df['X2'])}
        case 'BK':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': np.ones(X_size)}
        case 'CA':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': 2 * result_df['X2'], 'B3_true': np.exp(result_df['X3'])}
        case 'MA':
            B_true = {
                'B1_true': 0.2 * np.exp(0.2 * result_df['X1']),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': -2 * result_df['X3'],
                'B4_true': np.zeros(X_size)
            }
        case 'MB':
            B_true = {
                'B1_true': np.where(result_df['X1'] > 0,
                                    1 / (2 * np.sqrt(np.abs(result_df['X1']))),
                                    -1 / (2 * np.sqrt(np.abs(result_df['X1'])))),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': 1 / (result_df['X3'] + 7),
                'B4_true': np.zeros(X_size)
            }
        case 'MC':
            # y = X1*X2 + X3^2 + eps
            B_true = {
                'B1_true': result_df['X2'],                      # d/dX1: X2
                'B2_true': result_df['X1'],                      # d/dX2: X1
                'B3_true': 2 * result_df['X3'],                  # d/dX3: 2*X3
                'B4_true': np.zeros(X_size)                      # d/dX4: 0
            }
        case 'MD':
            # y = sin(X1) + cos(X2) + X3*X4 + eps
            B_true = {
                'B1_true': np.cos(result_df['X1']),              # d/dX1: cos(X1)
                'B2_true': -np.sin(result_df['X2']),             # d/dX2: -sin(X2)
                'B3_true': result_df['X4'],                      # d/dX3: X4
                'B4_true': result_df['X3']                       # d/dX4: X3
            }
        case 'ME':
            # y = X1 + X1*X2 + X3^2 + sin(X3*X4) + eps
            B_true = {
                'B1_true': 1 + result_df['X2'],                          # d/dX1: 1 + X2
                'B2_true': result_df['X1'],                              # d/dX2: X1
                'B3_true': 2*result_df['X3']
                        + result_df['X4']*np.cos(result_df['X3']*result_df['X4']),  # d/dX3: 2*X3 + X4*cos(X3*X4)
                'B4_true': result_df['X3']*np.cos(result_df['X3']*result_df['X4'])     # d/dX4: X3*cos(X3*X4)
            }
        case 'MF':
            # y = log(1+|X1|) + X2^3 + X2*X3 + eps
            B_true = {
                'B1_true': np.sign(result_df['X1']) / (1 + np.abs(result_df['X1'])),   # d/dX1: sign(X1)/(1+|X1|)
                'B2_true': 3 * result_df['X2']**2 + result_df['X3'],                   # d/dX2: 3*X2^2 + X3
                'B3_true': result_df['X2'],                                             # d/dX3: X2
                'B4_true': np.zeros(X_size)                                             # d/dX4: 0
            }
        case 'MG':
            # y = X1/(1+X2^2) + exp(-X3^2) + X4 + eps
            B_true = {
                'B1_true': 1 / (1 + result_df['X2']**2),                              # d/dX1: 1/(1+X2^2)
                'B2_true': -2*result_df['X1']*result_df['X2']
                        / (1 + result_df['X2']**2)**2,                              # d/dX2: -2*X1*X2/(1+X2^2)^2
                'B3_true': -2*result_df['X3'] * np.exp(-result_df['X3']**2),          # d/dX3: -2*X3*exp(-X3^2)
                'B4_true': np.ones(X_size)                                             # d/dX4: 1
            }
        case 'MH':
            # y = X1*X3 + |X2 - X4| + X3^2 + eps
            B_true = {
                'B1_true': result_df['X3'],                                             # d/dX1: X3
                'B2_true': np.sign(result_df['X2'] - result_df['X4']),                 # d/dX2: sign(X2-X4)
                'B3_true': result_df['X1'] + 2*result_df['X3'],                        # d/dX3: X1 + 2*X3
                'B4_true': -np.sign(result_df['X2'] - result_df['X4'])                 # d/dX4: -sign(X2-X4)
            }
        case 'MI':
            # y = tanh(X1+X2) + X3*sin(X1) + eps
            _sech2 = 1 / np.cosh(result_df['X1'] + result_df['X2'])**2                # sech^2(X1+X2)
            B_true = {
                'B1_true': _sech2 + result_df['X3']*np.cos(result_df['X1']),          # d/dX1: sech^2(X1+X2) + X3*cos(X1)
                'B2_true': _sech2,                                                     # d/dX2: sech^2(X1+X2)
                'B3_true': np.sin(result_df['X1']),                                    # d/dX3: sin(X1)
                'B4_true': np.zeros(X_size)                                            # d/dX4: 0
            }
        case 'MJ':
            # y = X1^2*X2 + sqrt(|X3|) + X4^3 + eps
            B_true = {
                'B1_true': 2*result_df['X1']*result_df['X2'],                         # d/dX1: 2*X1*X2
                'B2_true': result_df['X1']**2,                                        # d/dX2: X1^2
                'B3_true': np.sign(result_df['X3'])
                        / (2*np.sqrt(np.abs(result_df['X3']))),                    # d/dX3: sign(X3)/(2*sqrt(|X3|))
                'B4_true': 3*result_df['X4']**2                                       # d/dX4: 3*X4^2
            }
        case 'MK':
            # y = exp(X1*X2) + X3 + log(1+X4^2) + eps
            _exp_x1x2 = np.exp(result_df['X1'] * result_df['X2'])
            B_true = {
                'B1_true': result_df['X2'] * _exp_x1x2,                              # d/dX1: X2*exp(X1*X2)
                'B2_true': result_df['X1'] * _exp_x1x2,                              # d/dX2: X1*exp(X1*X2)
                'B3_true': np.ones(X_size),                                           # d/dX3: 1
                'B4_true': 2*result_df['X4'] / (1 + result_df['X4']**2)              # d/dX4: 2*X4/(1+X4^2)
            }
        case 'ML':
            # y = X1*sin(X2) + X2*cos(X3) + X1*X3 + eps
            B_true = {
                'B1_true': np.sin(result_df['X2']) + result_df['X3'],                 # d/dX1: sin(X2) + X3
                'B2_true': result_df['X1']*np.cos(result_df['X2'])
                        + np.cos(result_df['X3']),                                  # d/dX2: X1*cos(X2) + cos(X3)
                'B3_true': -result_df['X2']*np.sin(result_df['X3'])
                        + result_df['X1'],                                          # d/dX3: -X2*sin(X3) + X1
                'B4_true': np.zeros(X_size)                                            # d/dX4: 0
            }
        case 'MM':
            # y = log(1+|X1|) + 0.2*X2^3 + X2*X3 + eps
            B_true = {
                'B1_true': np.sign(result_df['X1']) / (1 + np.abs(result_df['X1'])),   # d/dX1: sign(X1)/(1+|X1|)
                'B2_true': 0.2 * 3 * result_df['X2']**2 + result_df['X3'],              # d/dX2: 3*X2^2 + X3
                'B3_true': result_df['X2'],                                             # d/dX3: X2
                'B4_true': np.zeros(X_size)                                             # d/dX4: 0
            }
        case 'MN':
            # y = X1^2*X2 + sqrt(|X3|) + X4^3 + eps
            B_true = {
                'B1_true': 2*result_df['X1']*result_df['X2'],                         # d/dX1: 2*X1*X2
                'B2_true': result_df['X1']**2,                                        # d/dX2: X1^2
                'B3_true': np.sign(result_df['X3'])
                        / (2*np.sqrt(np.abs(result_df['X3']))),                    # d/dX3: sign(X3)/(2*sqrt(|X3|))
                'B4_true': 0.3 * 3 * result_df['X4']**2                                       # d/dX4: 3*X4^2
            }
        case 'MO':
            B_true = {
                'B1_true': 0.2 * np.exp(0.2 * result_df['X1']),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': -2 * 0.25 * result_df['X3'],
                'B4_true': np.zeros(X_size)
            }
        case 'MP':
            B_true = {
                'B1_true': 0.2 * np.exp(0.2 * result_df['X1']),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': -2 * result_df['X3'],
                'B4_true': np.zeros(X_size)
            }
        case 'MQ':
            bp1 = -1.0
            bp2 = 1.0
            B_true = {
                'B1_true': np.where(result_df['X1'] < bp1,  2.0,
                           np.where(result_df['X1'] < bp2, -1.5,  1.8)),
                'B2_true': np.where(result_df['X2'] < bp1,  1.5,
                           np.where(result_df['X2'] < bp2, -1.0,  0.5)),
                'B3_true': np.where(result_df['X3'] < bp1, -1.0,
                           np.where(result_df['X3'] < bp2,  2.0, -1.5)),
                'B4_true': np.where(result_df['X4'] < bp1,  0.5,
                           np.where(result_df['X4'] < bp2, -0.8,  1.2)),
            }
        case 'MR':
            # y = 2*np.log1p(x[:,0]+1) + np.sin(0.7*x[:,1]) + 0.5*(x[:,2]-1)**(1.4) + x[:,3] + eps
            B_true = {
                'B1_true': 2 / (result_df['X1'] + 1),                                                   # d/dX1: 2/(X1+1)
                'B2_true': 0.7 * np.cos(0.7 * result_df['X2']),                                              # d/dX2: 0.7*cos(0.7*X2)
                'B3_true': 0.3 * 3 * (result_df['X3']-1)**2 + 0.3 * (result_df['X3']-1),                     # d/dX3:
                'B4_true': np.ones(X_size)                                             # d/dX4: 1
            }
        case 'MS':
            # y = X1 + 2*X2 + 0.5*X3 + X4^2 + eps
            B_true = {
                'B1_true': np.ones(X_size),                                             # d/dX1: 1
                'B2_true': 2 * np.ones(X_size),                                             # d/dX2: 2
                'B3_true': 0.5 * np.ones(X_size),                                             # d/dX3: 0.5
                'B4_true': 2 * result_df['X4']                                               # d/dX4: 2*X4
            }
        case 'MT':
            # y = X1 + 2*X2 + 0.5*X3 + X4^2 + eps
            B_true = {
                'B1_true': np.ones(X_size),                                             # d/dX1: 1
                'B2_true': 2 * np.ones(X_size),                                             # d/dX2: 2
                'B3_true': 0.5 * np.ones(X_size),                                             # d/dX3: 0.5
                'B4_true': 2 * result_df['X4']                                               # d/dX4: 2*X4
            }
        case 'Z1':
            B_true = {'B1_true': 0.8 * result_df['X1']}
        case 'Z2':
            B_true = {
                'B1_true': 0.8 * result_df['X1'],
                'B2_true': 0.8 * result_df['X2']
            }
        case 'Z3':
            B_true = {
                'B1_true': 0.8 * result_df['X1'],
                'B2_true': 0.8 * result_df['X2'],
                'B3_true': 0.8 * result_df['X3']
            }
        case 'Z4':
            B_true = {
                'B1_true': 0.8 * result_df['X1'],
                'B2_true': 0.8 * result_df['X2'],
                'B3_true': 0.8 * result_df['X3'],
                'B4_true': 0.8 * result_df['X4']
            }
        case 'Z5':
            B_true = {
                'B1_true': 0.8 * result_df['X1'],
                'B2_true': 0.8 * result_df['X2'],
                'B3_true': 0.8 * result_df['X3'],
                'B4_true': 0.8 * result_df['X4'],
                'B5_true': 0.8 * result_df['X5']
            }
        case 'friedman1':
            b0 = 10 * np.cos(np.pi * X[:,0] * X[:,1]) * np.pi * X[:,1]
            b1 = 10 * np.cos(np.pi * X[:,0] * X[:,1]) * np.pi * X[:,0]
            b2 = 40 * (X[:,2] - 0.5)
            b3 = 10 * np.ones(X_size)
            b4 = 5 * np.ones(X_size)
            B_true = {'B1_true': b0, 'B2_true': b1, 'B3_true': b2, 'B4_true': b3, 'B5_true': b4}
            for i in range(5, X.shape[1]):
                B_true[f'B{i+1}_true'] = np.zeros(X_size)
        case _:
            raise ValueError(f"Unknown DGP: {dgp_name}")

    # include intercept - in most cases it is zero
    B_true['B0_true'] = np.zeros(X_size)

    # manually change B0_true for several cases
    if dgp_name == 'MP':
        B_true['B0_true'] = 5 * np.ones(X_size)
    if dgp_name == 'UK':
        B_true['B0_true'] = a_vector[0] * np.ones(X_size)

    return B_true


