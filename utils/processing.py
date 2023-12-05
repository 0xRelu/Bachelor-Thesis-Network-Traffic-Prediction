import numpy as np

def longResults():
    meanMse600 = 0.365216463804245
    meanMae600 = 0.0696251168847084

    meanMse1200 = 0.3706625998020172
    meanMae1200 = 0.07179203629493713

    meanMse1800 = 0.40694430470466614
    meanMae1800 = 0.08026187121868134

    meanMse2400 = 0.3935747444629669
    meanMae2400 = 0.07650556415319443

    mse600 = np.array([0.346714049577713, 0.0948558822274208])
    mse1200 = np.array([0.3568652272224426, 0.1110810711979866])
    mse1800 = np.array([0.39902934432029724, 0.13328176736831665])
    mse2400 = np.array([0.4494786858558655, 0.15729382634162903])

    mse96 = mse600[::2] / meanMse600
    mae96 = mse600[1::2] / meanMae600

    mse192 = mse1200[::2] / meanMse1200
    mae192 = mse1200[1::2] / meanMae1200

    mse336 = mse1800[::2] / meanMse1800
    mae336 = mse1800[1::2] / meanMae1800

    mse720 = mse2400[::2] / meanMse2400
    mae720 = mse2400[1::2] / meanMae2400

    m96 = [f'{round(item, 3):.3f}' for pair in zip(mse96.tolist(), mae96.tolist()) for item in pair]
    m192 = [f'{round(item, 3):.3f}' for pair in zip(mse192.tolist(), mae192.tolist()) for item in pair]
    m336 = [f'{round(item, 3):.3f}' for pair in zip(mse336.tolist(), mae336.tolist()) for item in pair]
    m720 = [f'{round(item, 3):.3f}' for pair in zip(mse720.tolist(), mae720.tolist()) for item in pair]

    print(" & ".join(m96))
    print(" & ".join(m192))
    print(" & ".join(m336))
    print(" & ".join(m720))


def mainResults():
    mse96 = np.array([1.406, 1.19, 1.829, 1.165, 5.542, 3.027, 1.109, 1.127])
    mse192 = np.array([1.475, 1.293, 2.912, 1.238, 5.342, 4.303, 1.306, 1.229])
    mse336 = np.array([1.545, 1.336, 2.589, 1.319, 6.122, 6.586, 1.278, 1.340])
    mse720 = np.array([2.971, 1.463, 1.758, 1.45, 9.589, 10.001, 1.387, 1.586])

    mae96 = np.array(
        [0.329, 1.19, 0.301, 1.829, 0.376, 1.165, 0.291, 5.542, 0.553, 3.027, 0.414, 1.109, 0.242, 1.127, 0.27])
    mae192 = np.array(
        [0.343, 1.293, 0.321, 2.912, 0.465, 1.238, 0.306, 5.342, 0.582, 4.303, 0.485, 1.306, 0.266, 1.229, 0.284])
    mae336 = np.array(
        [0.343, 1.293, 0.321, 2.912, 0.465, 1.238, 0.306, 5.342, 0.582, 4.303, 0.485, 1.306, 0.266, 1.229, 0.284])
    mae720 = np.array(
        [0.541, 1.463, 0.368, 1.758, 0.412, 1.45, 0.355, 9.589, 0.869, 10.001, 0.887, 1.387, 0.312, 1.586, 0.358])

    mae96 = mae96[::2]
    mae192 = mae192[::2]
    mae336 = mae336[::2]
    mae720 = mae720[::2]

    mse96 = mse96 / 1.1148239374160767
    mae96 = mae96 / 0.24735787510871887

    mse192 = mse192 / 1.1732815504074097
    mae192 = mae192 / 0.25901708006858826

    mse336 = mse336 / 1.246145486831665
    mae336 = mae336 / 0.2745407521724701

    mse720 = mse720 / 1.3889302015304565
    mae720 = mae720 / 0.31892338395118713

    m96 = [f'{round(item, 3):.3f}' for pair in zip(mse96.tolist(), mae96.tolist()) for item in pair]
    m192 = [f'{round(item, 3):.3f}' for pair in zip(mse192.tolist(), mae192.tolist()) for item in pair]
    m336 = [f'{round(item, 3):.3f}' for pair in zip(mse336.tolist(), mae336.tolist()) for item in pair]
    m720 = [f'{round(item, 3):.3f}' for pair in zip(mse720.tolist(), mae720.tolist()) for item in pair]

    print(" & ".join(m96))
    print(" & ".join(m192))
    print(" & ".join(m336))
    print(" & ".join(m720))


if __name__ == "__main__":
    longResults()





