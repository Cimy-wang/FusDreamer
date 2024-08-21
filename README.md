# RS-FusDreamer
Official implementation of "A label-efficient remote sensing world model for multimodal data fusion"

The corresponding code will be released.

## The prompts of the Houston 2013 dataset.

| Classes | Prompts |
|-------------|-------------|
|  grass healthy|A hyperspectral and lidar multimodal data of grass healthy|
|			   |The grass healthy is next to the road|
|               |The grass healthy is dark green|
|			   |The spectral value of grass healthy is higher than that of the grass stressed|
|grass stressed |A hyperspectral and lidar multimodal data of grass stressed|
|			   |The grass stressed is closer to the road and parking lots|
|			   |The grass stressed is pale green|
|			   |The shape of the grass stressed is irregular|
|grass synthetic|A hyperspectral and lidar multimodal data of grass synthetic|
|			|The grass synthetic is located inside the running track 				|
|           |The shape of grass synthetic is a fixed-size rectangle 				|
|           |The spectrak value transformation interval of grass synthetic is small |
|	trees  |A hyperspectral and lidar multimodal data of trees|				
|           | The trees beside road            |
|           | The trees appear as small circles|
|           | Trees are higher than grass      |
|soil  |A hyperspectral and lidar multimodal data of soil|	
|           | The sail is tan						|
|           | The shape of the soil is irregular 	|
|           | The surface of soil is not smooth 	|
|	water   |A hyperspectral and lidar multimodal data of water|	
|    		|The water has a smooth surface			|
|           | Trees grew along the water 			|
|           | The water appears dark blue or black |
|	residential|A hyperspectral and lidar multimodal data of residential|	
|     		| Residential are densely packed 				|
|           | Residential buildings appear as small blocks |
|           | There are trees near the residential 			|
|	commercial|A hyperspectral and lidar multimodal data of commercial|	
|    		| The shapes of commercial are inconsistent 					|
|           | Commercial appear as large blocks 							|
|           | There are parking lot 1 and parking lot 2 near the commercial |
|	road|A hyperspectral and lidar multimodal data of road|	
|    		| Trees grew along the road 					|
|           | The road appear as elongated strip shape 		|
|           | Roads are narrower than highways and railways |
|	highway|A hyperspectral and lidar multimodal data of highway|	
|			| The highway is strip-shaped 									|
|           | Cars on the highway are discontinuous							|
|           | Highways are wider than railways								|
|	railway|A hyperspectral and lidar multimodal data of railway|
|     		| The railway is strip-shaped 									|
|           | The curvature of the railway is smooth 						|
|           | Trains on the railway are continuous							|	
|	parking lot 1|A hyperspectral and lidar multimodal data of parking lot 1|
|           | The area of parking lot 1 is empty 							|
|           | The parking lot 1 is next to the road 						|
|           | the parking lot 1 is near buildings 							|	
|	parking lot 2|A hyperspectral and lidar multimodal data of parking lot 2|	
|     		| The colors of parking lot 2 are messed up 					|
|           | The parking lot 2 is next to the road 						|
|           | the parking lot 2 is near buildings 							|
|	tennis court|A hyperspectral and lidar multimodal data of tennis court|	
|    		| There is also a crimson running track next to the tennis court|
|           | The height is close to the running track						|
|           | Tennis court is a regular rectangle							|
|	running track|A hyperspectral and lidar multimodal data of running track|
|     		| The running track is an ellipse 								|
|           | The running track is crimson 									|
|       	| There is grass synthetic in the middle of the running track 	|	

## The prompts of the Houston 2018 dataset.

   | Classes | Prompts |
|-------------|-------------|
   | grass healthy | A hyperspectral and lidar multimodal data of grass healthy |
   |               | The grass healthy is next to the road |
   |               | The grass healthy is dark green |
   |               | The spectral value of grass healthy is higher than that of the grass stressed  |
   | grass stressed |A hyperspectral and lidar multimodal data of grass stressed |
   |               |The grass stressed is closer to the road and parking lots|
   |               |The grass stressed is pale green|
   |               |The shape of the grass stressed is irregular|
   | artificial turf | A hyperspectral and lidar multimodal data of artificial turf |
   |               |The artificial turf is located inside the running track|
   |               | The shape of artificial turf is a fixed-size rectangle|
   |			   |The spectral value transformation interval of artificial turf is small|
   | evergreen trees|A hyperspectral and lidar multimodal data of artificial turf |
   |               |The evergreen trees beside road|
   |			   |The evergreen trees appear as small circles|
   |			   |The evergreen trees is dark green|
   | deciduous trees|A hyperspectral and lidar multimodal data of deciduous trees|
   |               |The trees beside road|
   |			   |The trees appear as small circles|
   |			   |The deciduous trees is yellowish-brown|
   | bare earth |A hyperspectral and lidar multimodal data of bare earth|
   |			   |The bare earth is tan|
   |               |The shape of the bare earth is irregular|
   |			   | The surface of bare earth is not smooth|
   |  water  |A hyperspectral and lidar multimodal data of water|
   |               |The water has a smooth surface|
   |			   |Trees grew along the water|
   |			   |The water appears dark blue or black|
   | residential buildings |A hyperspectral and lidar multimodal data of residential buildings|
   |               |Residential buildings are densely packed|
   |			   |Residential buildings appear as small blocks|
   |			   |There are trees near the residential buildings|
   |non-residential buildings |A hyperspectral and lidar multimodal data of non-residential buildings|
   |               |The shapes of non-residential buildings are inconsistent|
   |			   |Non-residential buildings appear as large blocks|
   |			   |Both paved and unpaved parking lots near the non-residential buildings|
   |  roads   |A hyperspectral and lidar multimodal data of roads|
   |               |Trees grew along the road|
   |			   | The road appear as elongated strip shape|
   |			   |Roads are narrower than highways and railways|
   |  sidewalks     |A hyperspectral and lidar multimodal data of sidewalks|
   |               |Sidewalks are parallel to road and major thoroughfares|
   |			   |Sidewalks are located near roads, major thoroughfares, and buildings|
   |			   |The distribution of sidewalks is irregular|
   | crosswalks   |A hyperspectral and lidar multimodal data of crosswalks|
   |               |Crosswalks are located above the road and major thoroughfares|
   |			   |Crosswalks are perpendicular to roads and major thoroughfares |
   |			   |Crosswalks connect two sidewalks|
   | major thoroughfares |A hyperspectral and lidar multimodal data of major thoroughfares|
   |               |Major thoroughfares are border road and highway|
   |			   |Major thoroughfares are wider than road|
   |			   | Major thoroughfares are rarely bend |
   | highway    |A hyperspectral and lidar multimodal data of highway|
   |               |The highway is strip-shaped|
   |			   | The highway and major thoroughfares cross|
   |			   |The highway and railway do not cross|
   |  railway   |A hyperspectral and lidar multimodal data of railway|
   |               |The railway is strip-shaped|
   |			   |The curvature of the railway is smooth|
   |			   |Trains on the railway are continuous|
   |  paved parking lots |A hyperspectral and lidar multimodal data of paved parking lots|
   |               |The paved parking lots are strip shape|
   |			   |The paved parking lots are next to the road|
   |			   |The paved parking lots are near buildings|
   | unpaved parking lots |A hyperspectral and lidar multimodal data of unpaved parking lots|
   |			   | The colors of parking lot 2 are messed up|
   |               | Unpaved parking lots are next to the road |
   |			   | The area of unpaved parking lots is small |
   |  cars    |A hyperspectral and lidar multimodal data of cars|
   |			   |The cars are next to the paved parking lots |
   |               |The cars are next to the non-residential buildings|
   |			   |The cars are discontinuous|
   |  trains    |A hyperspectral and lidar multimodal data of trains|
   |			   |The trains are strip shape|
   |               |The trains are next to the railway|
   |			   |The trains are continuous|
   | stadium seats |A hyperspectral and lidar multimodal data of stadium seats|
   |			   |There is artificial turf in the middle of the stadium seats|
   |               | The height of stadium seats is much higher than that of artificial turf|
   |			   | The stadium seats are an ellipse|

   
## The prompts of the MUUFL dataset.
      | Classes | Prompts |
   |-------------|-------------|
   |  tree  |A hyperspectral and lidar multimodal data of tree|
   |			   |The trees beside road|
   |               |The trees appear as small circles|
   |			   |Trees are higher than grass|
   |   mostly grass   |A hyperspectral and lidar multimodal data of mostly grass|
   |			   |The mostly grass is next to the road|
   |               |The grass healthy is green|
   |			   |The spectral value of grass healthy is higher than that of the grass stressed|
   |  mixed ground surface |A hyperspectral and lidar multimodal data of mixed ground surface|
   |			   |The mixed ground surface is yellow and green|
   |               |The mixed ground surface appears next to the tree|
   |			   |The mixed ground surface appears next to the sidewalk|
   | dirt and sand  |A hyperspectral and lidar multimodal data of dirt and sand|
   |			   |The bare earth is tan|
   |               |The shape of the dirt and sand is irregular|
   |			   |The surface of dirt and sand is not smooth|
   |   road   |A hyperspectral and lidar multimodal data of road|
   |			   |Trees grew along the road|
   |               |The building and building shadow are next to road|
   |			   |The road appear as elongated strip shape|
   |   water     |A hyperspectral and lidar multimodal data of water|
   |			   |The water has a smooth surface|
   |               |Trees grew along the water |
   |			   |The water appears black|
   |  building shadow   |A hyperspectral and lidar multimodal data of building shadow|
   |			   |The building shadow next to buildings|
   |               |The building shadow appears black|
   |			   |The building shadow is behind the building to the right|
   | building|A hyperspectral and lidar multimodal data of building|
   |			   |Building is densely packed|
   |               |Building appears as small blocks|
   |			   |There are trees near the building|
   |  sidewalk   |A hyperspectral and lidar multimodal data of sidewalk|
   |			   |Sidewalk is parallel to road|
   |               |Sidewalk is located near roads and buildings|
   |			   |The distribution of sidewalks is irregular|
   | yellow curb |A hyperspectral and lidar multimodal data of yellow curb|
   |			   |Yellow curb is parallel to road|
   |               |Yellow curb is yellow|
   |			   |Yellow curb are located near roads and sidewalks|
   |  cloth panels  |A hyperspectral and lidar multimodal data of cloth panels|
   |			   |Cloth panels are four regular rectangles|
   |               |Cloth panels cover the mixed ground surface|
   |			   |Cloth panels are located next to the trees|


## The experimental result on the Houston 2013 dataset with larger training samples following Ref. [AM<sup>3</sup>Net [IEEE TCSVT'2022]](https://ieeexplore.ieee.org/document/9698196).
   ![My Image](./img/houston13-L.png.png)
