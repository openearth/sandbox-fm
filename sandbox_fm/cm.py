import pathlib

import matplotlib.colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import hex2color, rgb2hex
from numpy import nan, inf

from pycpt.load import gmtColormap


def make_cmap(colors, position=None, bit=False, name='my_colormap'):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            raise ValueError("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            raise ValueError("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
    return cmap


data_dir = pathlib.Path(__file__).parent.parent / 'data'

# Used to reconstruct the colormap in viscm
parameters = {'xp': [-4.3306605284902275, -12.121422910237584, -3.3568152307718151, 30.727770189372848],
              'yp': [-29.083194212576501, -5.2239844184752258, 33.729827490261556, 14.739844184752371],
              'min_Jp': 16.6504854368932,
              'max_Jp': 63.252427184466015}

cm_data = [[ 0.05054409, 0.04310382, 0.51367225],
           [ 0.0449096 , 0.05785594, 0.50607837],
           [ 0.03954311, 0.0698548 , 0.49879243],
           [ 0.03465857, 0.08005226, 0.4917966 ],
           [ 0.03043403, 0.08895965, 0.48507445],
           [ 0.02682678, 0.09688914, 0.4786108 ],
           [ 0.0237982 , 0.10404784, 0.47239159],
           [ 0.02131334, 0.11058139, 0.46640384],
           [ 0.01934049, 0.11659677, 0.46063545],
           [ 0.01785086, 0.12217522, 0.45507524],
           [ 0.01681823, 0.1273801 , 0.44971276],
           [ 0.01621871, 0.13226193, 0.44453831],
           [ 0.01603047, 0.13686168, 0.43954282],
           [ 0.01623356, 0.14121309, 0.43471785],
           [ 0.01680972, 0.14534434, 0.43005548],
           [ 0.01774219, 0.14927917, 0.42554832],
           [ 0.01901558, 0.15303783, 0.42118945],
           [ 0.02061576, 0.15663768, 0.41697237],
           [ 0.02252969, 0.16009379, 0.41289098],
           [ 0.02474538, 0.16341926, 0.40893956],
           [ 0.02725175, 0.16662558, 0.40511273],
           [ 0.03003856, 0.16972288, 0.40140542],
           [ 0.03309635, 0.17272014, 0.39781285],
           [ 0.03641635, 0.17562532, 0.39433055],
           [ 0.03999041, 0.17844553, 0.39095443],
           [ 0.0436778 , 0.18118721, 0.38768029],
           [ 0.04738993, 0.18385612, 0.38450436],
           [ 0.05112041, 0.18645743, 0.38142305],
           [ 0.0548593 , 0.18899588, 0.37843299],
           [ 0.05859877, 0.19147571, 0.37553096],
           [ 0.0623327 , 0.19390084, 0.37271393],
           [ 0.06605629, 0.19627482, 0.36997903],
           [ 0.0697658 , 0.19860088, 0.36732352],
           [ 0.07345836, 0.20088203, 0.36474482],
           [ 0.07713176, 0.20312099, 0.36224047],
           [ 0.08078437, 0.20532029, 0.35980814],
           [ 0.08441494, 0.20748226, 0.35744561],
           [ 0.08802263, 0.20960904, 0.35515077],
           [ 0.09160682, 0.21170265, 0.35292161],
           [ 0.09516715, 0.21376491, 0.35075623],
           [ 0.09870343, 0.21579755, 0.34865282],
           [ 0.10221561, 0.21780216, 0.34660965],
           [ 0.10570376, 0.21978023, 0.34462507],
           [ 0.10916801, 0.22173316, 0.34269751],
           [ 0.1126086 , 0.22366224, 0.34082548],
           [ 0.11602579, 0.22556868, 0.33900756],
           [ 0.1194199 , 0.22745362, 0.33724239],
           [ 0.12279128, 0.22931813, 0.33552867],
           [ 0.12614028, 0.23116321, 0.33386517],
           [ 0.12946728, 0.23298982, 0.33225072],
           [ 0.13277267, 0.23479883, 0.33068419],
           [ 0.13605683, 0.23659109, 0.32916451],
           [ 0.13932016, 0.2383674 , 0.32769065],
           [ 0.14256303, 0.2401285 , 0.32626162],
           [ 0.14578582, 0.24187511, 0.3248765 ],
           [ 0.14898889, 0.24360791, 0.32353439],
           [ 0.15217258, 0.24532754, 0.32223445],
           [ 0.15533723, 0.24703461, 0.32097586],
           [ 0.15848315, 0.2487297 , 0.31975786],
           [ 0.16161064, 0.25041339, 0.3185797 ],
           [ 0.16471998, 0.2520862 , 0.31744067],
           [ 0.16781143, 0.25374866, 0.3163401 ],
           [ 0.1708852 , 0.25540125, 0.31527734],
           [ 0.17394153, 0.25704447, 0.31425176],
           [ 0.17698058, 0.25867878, 0.31326276],
           [ 0.18000252, 0.26030463, 0.31230978],
           [ 0.18300747, 0.26192246, 0.31139224],
           [ 0.18599553, 0.26353272, 0.31050962],
           [ 0.18896677, 0.26513581, 0.30966139],
           [ 0.19192121, 0.26673218, 0.30884702],
           [ 0.19485886, 0.26832223, 0.30806602],
           [ 0.19777967, 0.26990639, 0.30731788],
           [ 0.20068357, 0.27148507, 0.30660209],
           [ 0.20357041, 0.27305869, 0.30591814],
           [ 0.20644005, 0.27462768, 0.3052655 ],
           [ 0.20929227, 0.27619248, 0.30464362],
           [ 0.21212679, 0.27775352, 0.30405193],
           [ 0.21494336, 0.27931127, 0.30348973],
           [ 0.21774156, 0.28086619, 0.30295643],
           [ 0.22052098, 0.28241877, 0.30245129],
           [ 0.22328115, 0.28396952, 0.3019735 ],
           [ 0.22602155, 0.28551897, 0.30152216],
           [ 0.22874162, 0.28706766, 0.30109622],
           [ 0.23144074, 0.28861618, 0.3006945 ],
           [ 0.23411828, 0.2901651 , 0.30031565],
           [ 0.23677358, 0.29171506, 0.29995811],
           [ 0.23940599, 0.29326669, 0.29962008],
           [ 0.24201487, 0.29482065, 0.29929948],
           [ 0.24459963, 0.2963776 , 0.29899392],
           [ 0.24715979, 0.29793821, 0.29870065],
           [ 0.24969501, 0.29950312, 0.29841652],
           [ 0.25220512, 0.30107295, 0.29813794],
           [ 0.25469026, 0.30264828, 0.29786087],
           [ 0.25715087, 0.30422959, 0.29758079],
           [ 0.25958783, 0.30581725, 0.29729268],
           [ 0.26200254, 0.30741152, 0.29699108],
           [ 0.26439695, 0.30901245, 0.29667009],
           [ 0.26677368, 0.3106199 , 0.29632349],
           [ 0.26913605, 0.31223349, 0.2959448 ],
           [ 0.2714881 , 0.31385255, 0.29552749],
           [ 0.27383455, 0.31547617, 0.29506506],
           [ 0.27618078, 0.31710312, 0.29455128],
           [ 0.27853272, 0.31873193, 0.29398036],
           [ 0.28089669, 0.32036088, 0.29334716],
           [ 0.28327923, 0.32198804, 0.29264727],
           [ 0.28568694, 0.32361136, 0.29187723],
           [ 0.28812624, 0.32522869, 0.29103452],
           [ 0.29060321, 0.32683787, 0.29011763],
           [ 0.29312343, 0.32843678, 0.28912601],
           [ 0.29569185, 0.3300234 , 0.28806004],
           [ 0.29831275, 0.33159582, 0.28692067],
           [ 0.30098959, 0.33315234, 0.28570981],
           [ 0.30372505, 0.33469142, 0.28442994],
           [ 0.30652108, 0.33621173, 0.28308384],
           [ 0.30937899, 0.33771215, 0.28167459],
           [ 0.31229944, 0.33919172, 0.28020541],
           [ 0.31528258, 0.34064966, 0.27867963],
           [ 0.31832812, 0.34208535, 0.27710056],
           [ 0.32143539, 0.34349829, 0.27547143],
           [ 0.32460345, 0.34488811, 0.27379538],
           [ 0.32783112, 0.34625453, 0.27207542],
           [ 0.3311171 , 0.34759734, 0.27031439],
           [ 0.33445994, 0.34891638, 0.26851499],
           [ 0.33785815, 0.35021155, 0.26667974],
           [ 0.34131022, 0.35148279, 0.26481101],
           [ 0.34481461, 0.35273005, 0.26291102],
           [ 0.34837031, 0.35395318, 0.26098098],
           [ 0.35197538, 0.35515227, 0.25902369],
           [ 0.35562839, 0.35632729, 0.25704104],
           [ 0.35932795, 0.35747823, 0.25503478],
           [ 0.36307272, 0.35860508, 0.2530066 ],
           [ 0.36686144, 0.3597078 , 0.25095812],
           [ 0.37069286, 0.36078637, 0.2488909 ],
           [ 0.37456584, 0.36184075, 0.24680645],
           [ 0.37847923, 0.36287088, 0.24470627],
           [ 0.38243255, 0.36387655, 0.24259091],
           [ 0.38642442, 0.36485779, 0.24046239],
           [ 0.39045368, 0.36581458, 0.23832247],
           [ 0.39451938, 0.36674684, 0.23617259],
           [ 0.3986206 , 0.36765451, 0.23401423],
           [ 0.40275647, 0.36853748, 0.23184884],
           [ 0.40692613, 0.36939567, 0.22967794],
           [ 0.41112923, 0.37022883, 0.22750234],
           [ 0.4153649 , 0.37103686, 0.22532372],
           [ 0.41963193, 0.37181982, 0.2231443 ],
           [ 0.42392952, 0.37257761, 0.22096573],
           [ 0.42825692, 0.37331014, 0.21878969],
           [ 0.43261333, 0.37401733, 0.21661794],
           [ 0.43699834, 0.37469896, 0.21445181],
           [ 0.44141135, 0.37535488, 0.21229294],
           [ 0.44585105, 0.37598523, 0.210144  ],
           [ 0.45031666, 0.37658995, 0.20800698],
           [ 0.45480741, 0.377169  , 0.20588391],
           [ 0.45932252, 0.37772233, 0.2037769 ],
           [ 0.46386176, 0.37824968, 0.2016874 ],
           [ 0.46842379, 0.37875126, 0.19961834],
           [ 0.47300768, 0.37922712, 0.19757218],
           [ 0.47761261, 0.37967728, 0.19555128],
           [ 0.48223774, 0.38010179, 0.19355806],
           [ 0.48688245, 0.38050058, 0.19159473],
           [ 0.49154567, 0.38087383, 0.18966409],
           [ 0.49622638, 0.3812217 , 0.1877689 ],
           [ 0.50092369, 0.38154431, 0.18591181],
           [ 0.50563671, 0.3818418 , 0.18409549],
           [ 0.51036447, 0.38211433, 0.18232271],
           [ 0.51510595, 0.38236216, 0.18059633],
           [ 0.51986022, 0.38258549, 0.17891911],
           [ 0.52462637, 0.38278453, 0.17729382],
           [ 0.52940344, 0.38295953, 0.17572322],
           [ 0.53419018, 0.38311094, 0.17421043],
           [ 0.53898559, 0.38323907, 0.17275821],
           [ 0.54378896, 0.38334413, 0.17136906],
           [ 0.54859935, 0.38342642, 0.17004563],
           [ 0.55341586, 0.3834863 , 0.16879052],
           [ 0.55823723, 0.38352431, 0.1676066 ],
           [ 0.56306204, 0.38354113, 0.16649674],
           [ 0.56789011, 0.38353675, 0.16546266],
           [ 0.57272058, 0.38351159, 0.16450661],
           [ 0.5775526 , 0.38346603, 0.16363072],
           [ 0.58238534, 0.3834005 , 0.16283698],
           [ 0.58721731, 0.38331586, 0.1621278 ],
           [ 0.59204757, 0.38321265, 0.16150494],
           [ 0.59687614, 0.38309083, 0.16096934],
           [ 0.60170232, 0.38295084, 0.16052236],
           [ 0.60652544, 0.38279309, 0.16016519],
           [ 0.61134486, 0.38261801, 0.15989884],
           [ 0.61615997, 0.38242602, 0.15972412],
           [ 0.62096987, 0.38221775, 0.15964186],
           [ 0.62577313, 0.38199422, 0.15965293],
           [ 0.63057048, 0.38175502, 0.15975686],
           [ 0.63536148, 0.38150051, 0.15995373],
           [ 0.64014572, 0.38123105, 0.1602434 ],
           [ 0.64492284, 0.38094697, 0.16062558],
           [ 0.64969251, 0.3806486 , 0.1610998 ],
           [ 0.65445444, 0.38033625, 0.16166542],
           [ 0.65920839, 0.3800102 , 0.16232165],
           [ 0.66395413, 0.37967073, 0.16306758],
           [ 0.6686915 , 0.37931806, 0.16390213],
           [ 0.67342034, 0.37895244, 0.16482412],
           [ 0.67814057, 0.37857405, 0.16583225],
           [ 0.68285211, 0.37818307, 0.16692513],
           [ 0.68755493, 0.37777965, 0.16810128],
           [ 0.69224902, 0.37736391, 0.16935914],
           [ 0.69693442, 0.37693596, 0.17069709],
           [ 0.70161118, 0.37649587, 0.17211347],
           [ 0.70627939, 0.37604369, 0.17360657],
           [ 0.71093918, 0.37557943, 0.17517465],
           [ 0.71559069, 0.37510308, 0.17681596],
           [ 0.7202341 , 0.37461461, 0.17852873],
           [ 0.72486958, 0.37411396, 0.1803112 ],
           [ 0.72949738, 0.37360104, 0.18216162],
           [ 0.73411771, 0.37307572, 0.18407824],
           [ 0.73873086, 0.37253786, 0.18605936],
           [ 0.74333709, 0.37198729, 0.18810328],
           [ 0.74793671, 0.37142381, 0.19020835],
           [ 0.75253003, 0.37084717, 0.19237296],
           [ 0.75711739, 0.37025713, 0.19459554],
           [ 0.76169912, 0.3696534 , 0.19687456],
           [ 0.76627278, 0.36903842, 0.19920754],
           [ 0.77084102, 0.36840967, 0.20159373],
           [ 0.77540452, 0.36776653, 0.20403184],
           [ 0.77996366, 0.3671086 , 0.20652056],
           [ 0.78451884, 0.36643547, 0.2090586 ],
           [ 0.7890705 , 0.3657467 , 0.21164476],
           [ 0.79361903, 0.3650418 , 0.21427789],
           [ 0.7981607 , 0.36432471, 0.21695471],
           [ 0.80269943, 0.3635913 , 0.21967582],
           [ 0.80723611, 0.36284057, 0.22244044],
           [ 0.81177118, 0.36207193, 0.22524764],
           [ 0.8163051 , 0.36128478, 0.22809653],
           [ 0.82083546, 0.36048169, 0.23098454],
           [ 0.82536251, 0.35966232, 0.23391058],
           [ 0.82988955, 0.35882287, 0.23687564],
           [ 0.83441706, 0.3579626 , 0.23987905],
           [ 0.83894552, 0.35708075, 0.24292018],
           [ 0.84346935, 0.35618369, 0.24599421],
           [ 0.84799407, 0.35526482, 0.24910394],
           [ 0.85252093, 0.35432239, 0.25224942],
           [ 0.85705042, 0.35335551, 0.25543023],
           [ 0.86157709, 0.35237061, 0.2586414 ],
           [ 0.86610537, 0.35136206, 0.26188541],
           [ 0.8706375 , 0.35032673, 0.26516327],
           [ 0.87517397, 0.34926359, 0.26847471],
           [ 0.87970675, 0.34818262, 0.27181246],
           [ 0.88424459, 0.34707228, 0.2751829 ],
           [ 0.888788  , 0.3459314 , 0.2785859 ],
           [ 0.89333304, 0.34476474, 0.28201749],
           [ 0.89787967, 0.34357202, 0.28547692],
           [ 0.90243312, 0.34234568, 0.2889682 ],
           [ 0.90699224, 0.34108668, 0.29248981],
           [ 0.91155077, 0.33980327, 0.29603549],
           [ 0.91611737, 0.33848278, 0.29961257],
           [ 0.92069238, 0.33712395, 0.30322096],
           [ 0.92526591, 0.33574047, 0.30685088],
           [ 0.92984877, 0.33431601, 0.31051201],
           [ 0.93444147, 0.33284891, 0.31420449]]

terrajet = ListedColormap(cm_data, name=__file__)
# colors = [hex2color(hex) for hex in ('#2F3360', '#00C1FF', '#366032', '#BAA838', '#BA5C21')]
colors = [hex2color(hex) for hex in ('#111160', '#00FFFF', '#11DD00', '#FFFF00', '#DD3322', '#FF0044', '#DDDDDD')]
terrajet2 = make_cmap(colors)
colombia = gmtColormap(str(data_dir / 'colombia.cpt'))
transparent_water = matplotlib.colors.LinearSegmentedColormap.from_list(
    'transparent_water',
    [
        (0, (0.7, 0.7, 0.9, 0.3)),
        (0.1, (0.6, 0.6, 0.9, 0.7)),
        (0.2, (0.4, 0.5, 0.9, 0.8)),
        (0.7, (0.1, 0.2, 1.0, 0.9)),
        (1.0, (0.1, 0.2, 0.8, 0.9))
    ]
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from viscm import viscm
        viscm(test_cm)
    except ImportError:
        print("viscm not found, falling back on simple display")
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                   cmap=test_cm)
    plt.show()
