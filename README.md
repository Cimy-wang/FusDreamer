# RS-FusDreamer
Official implementation of "A label-efficient remote sensing world model for multimodal data fusion"

   | Classes | Prompts |
   | — | — |
   | grass healthy | A hyperspectral and lidar multimodal data of grass healthy |
   |               | The grass healthy is next to the road |
   |               | The grass healthy is dark green |
   |               | The spectral value of grass healthy is higher than that of the grass stressed  |
   | — | — |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |
   |   |   |



   
   % \begin{table}[htbp]
%   \centering
%   \caption{Text description for the Houston18 dataset.}
%     \begin{tabular}{|c|p{5.4 cm}|}
%     \hline
%     \multicolumn{1}{|c|}{Class name} & \multicolumn{1}{c|}{Text description} \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{}} & The grass healthy is next to the road \\
%     \multicolumn{1}{|c|}{} &  \\
%     \multicolumn{1}{|c|}{} &   \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{grass stressed}} & The grass stressed is closer to the road and parking lots \\
%     \multicolumn{1}{|c|}{} & The grass stressed is pale green \\
%     \multicolumn{1}{|c|}{} & The shape of the grass stressed is irregular \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{artificial turf}} & The artificial turf is located inside the running track \\
%     \multicolumn{1}{|c|}{} & The shape of artificial turf is a fixed-size rectangle \\
%     \multicolumn{1}{|c|}{} & The spectrak value transformation interval of artificial turf is small \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{evergreen trees}} & The evergreen trees beside road \\
%     \multicolumn{1}{|c|}{} & The evergreen trees appear as small circles \\
%     \multicolumn{1}{|c|}{} & The evergreen trees is dark green \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{deciduous trees}} & The trees beside road \\
%     \multicolumn{1}{|c|}{} & The trees appear as small circles \\
%     \multicolumn{1}{|c|}{} & The deciduous trees is yellowish-brown \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{bare earth}} & The bare earth is tan \\
%     \multicolumn{1}{|c|}{} & The shape of the bare earth is irregular \\
%     \multicolumn{1}{|c|}{} & The surface of bare earth is not smooth \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{water}} & The water has a smooth surface \\
%     \multicolumn{1}{|c|}{} & Trees grew along the water \\
%     \multicolumn{1}{|c|}{} & The water appears dark blue or black \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{residential buildings}} & Residential buildings are densely packed \\
%     \multicolumn{1}{|c|}{} & Residential buildings appear as small blocks \\
%     \multicolumn{1}{|c|}{} & There are trees near the residential buildings \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{non-residential buildings}} & The shapes of non-residential buildings are inconsistent \\
%     \multicolumn{1}{|c|}{} & Non-residential buildings appear as large blocks \\
%     \multicolumn{1}{|c|}{} & \textcolor[rgb]{ 1,  0,  0}{Both paved and unpaved parking lots near the non-residential buildings} \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{roads}} & Trees grew along the road \\
%     \multicolumn{1}{|c|}{} & The road appear as elongated strip shape \\
%     \multicolumn{1}{|c|}{} & Roads are narrower than highways and railways \\
%     \hline
%     \multirow{3}[2]{*}{sidewalks} & Sidewalks are parallel to road and major thoroughfares \\
%           & Sidewalks are located near roads, major thoroughfares, and buildings \\
%           & The distribution of sidewalks is irregular \\
%     \hline
%     \multirow{3}[2]{*}{crosswalks} & Crosswalks are lcoated above the road and major thoroughfares \\
%           & Crosswalks are perpendicular to roads and major thoroughfares \\
%           & Crosswalks connect two sidewalks \\
%     \hline
%     \multicolumn{1}{|c|}{\multirow{3}[2]{*}{major thoroughfares}} & Major thoroughfares are border road and highway \\
%     \multicolumn{1}{|c|}{} & Major thoroughfares are wider than road \\
%     \multicolumn{1}{|c|}{} & Major thoroughfares are rarely bend \\
%     \hline
%     \multirow{3}[2]{*}{highway} & The highway is strip-shaped \\
%           & The highway and major thoroughfares cross \\
%           & The highway and railway do not cross \\
%     \hline
%     \multirow{3}[2]{*}{railway} & The railway is strip-shaped \\
%           & The curvature of the railway is smooth \\
%           & Trains on the railway are continuous \\
%     \hline
%     \multirow{3}[2]{*}{paved parking lots} & The paved parking lots are strip shape \\
%           & The paved parking lots are next to the road \\
%           & The paved parking lots are near buildings \\
%     \hline
%     \multirow{3}[2]{*}{unpaved parking lots} & The colors of parking lot 2 are messed up \\
%           & Unpaved parking lots are next to the road \\
%           & The erea of unpaved parking lots is small \\
%     \hline
%     \multirow{3}[2]{*}{cars} & The cars are next to the paved parking lots \\
%           & The cars are next to the non-residential buildings \\
%           & The cars are discontinuous \\
%     \hline
%     \multirow{3}[2]{*}{trains} & The trains are strip shape \\
%           & The trains are next to the railway \\
%           & The trains are continuous \\
%     \hline
%     \multirow{3}[2]{*}{stadium seats} & There is artificial turf in the middle of the stadium seats \\
%           & The height of stadium seats is much higher than that of artificial turf \\
%           & The stadium seats are an ellipse \\
%     \hline
%     \end{tabular}%
%   \label{tab:promptHouston18}%
% \end{table}%
