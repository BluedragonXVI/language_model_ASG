# What is SLAM?

SLAM, short for *simultaneous localization and mapping*, is the computer vision task of efficiently and accurately creating a map of an agent's surroundings, while at the same time determining the agent's location in said map. The tasks of localization and mapping were originally studied separately, but it soon became apparent that both were dependent on the other. Localization tries to answer "where am I?", whereas mapping answers "what does the world look like?"

**Localization:** Determination of the agent's position and orientation (pose) within the map

**Mapping:** Integration of partial observations of surroundings into one consistent model

SLAM is considered a difficult problem for multiple reasons:

* A map is needed for localization
* Localization is required for map building
* Both are usually unknown
* Incorrectly associating a landmark with a location can be disasterous (e.g. high speed drone crashing into an object/civilian)
* Although solved for some sensor types and map types (sonar, laser, 2D-maps, small static environments) *visual* SLAM (where camera/images are the only sensor/input) on large dynamic areas is still an open research problem

