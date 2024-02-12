
best_prompt_weight = {}
best_prompt_weight['scanobjectnn_rn50_test_prompts'] = ['This view shows a bag from the side.', 'The left view depth map would show the bin as being relatively close to the viewer, while the right view depth map would show the bin as being further away from the viewer.', 'In an obscure depth map of a box, the top and bottom surfaces of the box would appear close together, while the sides of the box would appear further apart.', 'The cabinet will be a darker color than the surrounding area.', "The chair's depth map is foggy and hard to decipher.", 'The depth map would show the desk in great detail, revealing all of its nooks and crannies.', 'A grayscale or white depth map of a display 3D model is a two-dimensional array of numbers that represent the distance of each point on the model from the viewer.', 'A caption of a white depth photo of a door could be "a door leading to a light-filled room.', 'This is a gray or white depth map of a shelf.', 'The table would appear to be floating above the ground, with a slight shadow underneath it.', 'A white heightmap on a black background of a bed would look like a white pillow on a black bed.', 'A white heightmap on a black background of a pillow would look like a mountainscape in the night.', 'A depth map of a sink viewed from the side would look like a rectangle with a curved bottom.', 'This is an unclear depth map of a sofa against a black background.', ' The heightmap would look like a white bowl in the center of a black background.']
best_prompt_weight['scanobjectnn_rn101_test_prompts'] = ['This is an image of an obscured depth map of a bag.', 'An diffuse grayscale depth map of a oblique rough a bin model.', 'This depth map of a 3D model of a box shows the relative distance of various points on the surface of the box from the viewer.', 'An obscure grayscale depth map of a inclined rough a cabinet model.', 'A chair in a grayscale map would likely appear as a dark gray blob.', 'This image is a grayscale depth map of a desk.', 'A dull monochrome depth map of a slanted uneven a show model.', 'An unclear grayscale depth map of a slanted rough a door model.', 'An indistinct grayscale depth map of a slanted rough a shelf model.', 'An unintelligible grayscale depth map of a slanted rough a table model.', 'This depth map is either grayscale or white and shows a bed.', 'If a pillow is present in a grayscale map, it will appear as a dark gray or black object.', 'A crude 3D model of an slanted a sink composed of gray unidentifiable balls.', 'An unclear grayscale depth map of a slanted rough a sofa model.', 'An inclined toilet composed of gray, obscure balls that is 3D and rough.']
best_prompt_weight['scanobjectnn_vit_b16_test_prompts'] = ['The depth map of the bag is dark, shadowy, and mysterious.', 'This is a monochrome depth map of a bin.', 'This is a 3D model of a box with an obscure depth map.', 'A side view depth map of a cabinet might look like a rectangle with a few smaller rectangles inside of it.', 'The chair would be a light gray color on the grayscale map.', 'The desk would appear as a rectangle on the grayscale map.', 'This is a depth map of a display in grayscale or white.', 'A side view depth map of a door would look like a rectangle with a line down the middle.', 'A white heightmap in a black background of a shelf would appear as a white relief map on a black background.', 'A grayscale or white depth map of a table 3D model would show the table as a white object with varying shades of gray depending on the distance of the table from the viewer.', 'A side view depth map of a bed looks like a rectangle with a smaller rectangle on top of it.', 'The white heightmap would look like a pillow with a black background.', 'This is a depth map of a 3D model of a sink.', 'The sofa would be the lightest color on the grayscale map.', 'The image would be a white depth map of a 3D model of a toilet.']
best_prompt_weight['scanobjectnn_vit_b32_test_prompts'] = ['This is a depth map of a bag.', 'An obscure depth map of a bin 3D model would show the bin in great detail, revealing all of its nooks and crannies.', 'This depth map shows the relative depth of objects in a simple scene featuring a single box.', 'The white heightmap would be in the shape of a cabinet, with a black background.', "The chair's depth map is foggy and hard to decipher.", 'The depth map is porous and obscure, with a black background.', 'A side view depth map of a display looks like a two-dimensional image with various Shades of gray.', 'This is a door rendered in grayscale or white.', 'The depth map of a white shelf would appear as a flat, even surface with no shadows or highlights.', 'An obscure depth map of a table would show the table in great detail, but the background would be very blurry.', 'The heightmap is a white bed in a black background.', 'This heightmap depicts a pillow, which is a soft and comfortable object often used for sleeping.', 'A rough 3D model of a slanted sink made of gray obscure balls.', 'An unclear depth map in shades of gray of a slanted couch model.', 'I found a toilet depth map online.']
best_prompt_weight['scanobjectnn_rn50_test_weights'] = [0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25]
best_prompt_weight['scanobjectnn_rn101_test_weights'] = [0.50, 0.50, 0.50, 0.75, 0.50, 1.00, 1.00, 0.75, 0.25, 0.25]
best_prompt_weight['scanobjectnn_vit_b16_test_weights'] = [0.75, 0.75, 0.75, 0.5, 0.5, 1.0, 0.75, 1.0, 0.25, 0.25]
best_prompt_weight['scanobjectnn_vit_b32_test_weights'] = [0.75, 0.75, 0.75, 0.5, 0.75, 0.75, 0.5, 1.0, 0.5, 0.75]


best_prompt_weight['modelnet40_rn50_test_prompts'] = ['This is an airplane with a black background and a porous, obscure depth map.', 'A vague greyscale depth map of a slanted coarse a bathtub model.', 'An blurred greyscale depth map of a slanted coarse a bed model.', 'An obscure grayscale depth map of an inclined rough bench 3D model.', 'This is a depth map of a bookshelf rendered in shades of gray or white.', 'The left or right view of an off-white a bottle would appear as a dark shape against a lighter background.', 'On a grayscale map, a bowl would appear as a dark area surrounded by lighter areas.', 'If a car is present on a grayscale map, it will appear as a dark object.', 'A chair on a grayscale map would likely appear as a dark object with a light object on top of it.', 'A depth map of a cone would look like a cone shaped object with varying shades of gray.', 'A cup can be identified in a grayscale map by its darker color.', 'An obscure grayscale depth map of an inclined rough curtain 3D model.', 'A murky greyscale depth map of a slanted textured a desk model.', 'An obscure grayscale depth map of an inclined rough door 3D model.', 'In a grayscale map, a dresser would likely appear as a gray rectangle.', 'A white and porous depth map of a flower pot would have a light color with a bumpy texture.', 'A gentle, unassuming light grey heightmap floats in the center of a dark, inky black background.', 'The sentence is describing an image of a guitar that is in grayscale and is not very clear.', 'A grayscale or white depth map of a keyboard 3D model would represent the keyboard in different shades of gray, with the keys themselves being the lightest shade and the deepest part of the keyboard being the darkest shade.', 'A depth map of a lamp would likely show the different parts of the lamp in different shades of gray, with the lightest parts being the farthest away from the viewer and the darkest parts being the closest.', 'This is a depth map of a laptop in grayscale or white.', 'An unclear depth map in shades of gray of a mantel model that is slanted and coarse.', 'An unnoticeable grayscale depth map of a slanted rough a monitor model.', 'This is a description of an image, specifically a grayscale depth map of a white nightstand that is inclined.', 'This person is standing in an empty room, and their outline is shown in stark relief against the featureless background.', 'If you are looking at a map in grayscale, pianos will appear as a medium-dark gray.', 'The following is a grayscale or white depth map of a plant.', 'An obscure grayscale depth map of an inclined rough radio 3D model.', 'A white depth map of a range hood 3D model would look like a white silhouette of the range hood.', 'A side view depth map of a sink would look like a two-dimensional image of the sink, with the different depths represented by different shades of gray.', 'The sofa 3D model is a an object that exists in three dimensions.', 'This is a description of an image, specifically a grayscale depth map of a staircase.', 'One way to identify a stool from a grayscale map is to look for an area of high contrast.', 'A render of a table with four legs, a flat surface, and a slight lip around the edge.', 'If you are looking at a grayscale map, a tent would appear as a light gray circle.', 'The toilet 3D model would appear as a white object on a grayscale background.', 'An obscure grayscale depth map of an inclined rough tv stand 3D model.', 'This is a rough 3D model of a vase composed of gray, obscure balls, placed at an incline.', 'An obscure grayscale depth map of an inclined rough wardrobe 3D model.', 'An obscure grayscale depth map of an inclined rough xbox 3D model.']
best_prompt_weight['modelnet40_rn101_test_prompts'] = ['If you are looking at a grayscale map, you can identify an airplane by looking for a small cluster of pixels that are a different color than the surrounding pixels.', 'An obscure, grayscale depth map of an inclined, rough bathtub model.', 'A dim grayscale depth map of a steep textured a bed model.', 'The depth map shows the contours of a 3D model of a bench.', 'An unclear grayscale depth map of a slanted rough a bookshelf model.', 'The left view of an off-white bottle is the side of the bottle that is facing the left.', 'A small, round, white 3D model of a bowl.', 'An obscured grayscale depth map of an inclined rough car model.', 'This depth map is in grayscale or white, and it shows a chair.', 'The feature of an obscure depth map of a cone is that it has very little contrast and appears to be a flat surface.', 'If the map is a grayscale, the cup will be a dark object surrounded by a lighter area.', 'An unclear greyscale depth map of a slanted rough a curtain model.', 'This grayscale or white depth map is of a desk.', 'An obscure grayscale depth map of an inclined rough door 3D model.', 'An obscure grayscale depth map of an inclined rough a dresser which is white.', 'The flower pot would be a dark spot on the grayscale map.', 'An oblique grayscale depth map of a rough glass box model.', 'This depth map is either in grayscale or white, and it is of a guitar.', 'An unknown grayscale depth map of a slanted rough a keyboard model.', 'This is a simple heightmap of a lamp.', 'An obscure grayscale depth map of an inclined rough laptop 3D model.', 'An obscure grayscale depth map of an inclined rough mantel 3D model.', 'An unnoticeable grayscale depth map of a slanted rough a monitor model.', 'The observed depth map (Figure 1) is a 3D model of a night stand.', 'An unclear grayscale depth map of a slanted rough human model.', 'An obscure grayscale depth map of a inclined rough a piano model.', 'This depth map is either grayscale or white, depicting a plant.', 'An obscure grayscale depth map of an inclined rough radio 3D model.', 'An obscure grayscale depth map of an inclined rough range hood model.', 'An unclear grayscale depth map of a slanted rough a sink model.', 'A blurry gray and white depth map of a couch model at a slanted angle.', 'An unidentifiable grayscale depth map of a slanted rough a stairs model.', 'This is a depth map of either a gray or white stool.', 'A crude 3D model of an incline table made of drab, dull balls.', 'An ambiguous grayscale depth map of a tilted rough a tent model.', 'The grayscale depth map is an image of a toilet that is slightly tilted and has a rough surface.', 'An obscure grayscale depth map of an inclined rough tv stand 3D model.', 'A grayscale image of a vase with a light source shining from the left.', 'This is a greyscale depth map of a wardrobe.', 'An obtuse grayscale depth map of a slanted rough a xbox model.']
best_prompt_weight['modelnet40_vit_b16_test_prompts'] = ['A three-dimensional model of an airplane composed of gray, fuzzy balls.', 'A lumpy 3D model of a slanted bathtub made of dull gray balls.', 'A bed is typically represented by a rectangular shape on a grayscale map.', 'A bench can be identified from a grayscale map by its shape.', 'An unclear depth map in shades of gray of a bookshelf model at a slanted angle.', 'A bottle generally has a cylindrical shape and a narrow neck.', 'A 3D model of a bowl composed of gray balls that are difficult to see.', 'This is a depth map of a car 3D model, generated by depth sensing cameras.', 'The chair would be a dark object on the grayscale map.', 'A 3D model of a cone would look like a cone shape.', 'An obscure cup was found at the depth map.', 'There is less light in an obscure depth map of a curtain, so the features are not as clearly defined.', 'An obscure depth map of a desk would likely show a few objects on the desk in great detail, while the rest of the desk would be less detailed and more blurry.', 'This sketch depth map shows the door of a room.', 'A white heightmap in a black background of a dresser would look like a white rectangle in the center of the dresser with a black border.', 'A depth map of a flower pot can be identified by its shading.', 'A glass box will appear as a bright white region in a depth map.', 'I am looking at a 3D model of a guitar.', '3D render of a keyboard with heightmap.', 'There is an obscure depth map of a lamp.', 'The depth map of a laptop 3D model is a top-down view of the laptop, showing its various components in different colors.', 'The left or right view depth map of a white mantel would look like a white rectangle with some depth to it.', 'A grayscale image of a monitor.', 'A white, porous depth map of a night stand might look like a ghostly image of the furniture piece, with its contours and dimensions visible but slightly blurred.', 'A heightmap of a person, showing their height at different points along their body.', 'The piano would be the blackest object on the grayscale map.', 'This depth map is of a plant against a black background and is full of pores.', 'A typical radiolooks like a rectangular box with a handle on the top.', 'There is an obscure depth map of a range hood.', 'A depth map of a sink 3D model can be quite obscure, as the sink is often hidden behind other objects in a room.', 'This is a depth map of a 3D model of a sofa.', 'Thestairs3Dmodelhas a Depth Map that is quite Obscure .', 'A stool can vary in shape and size, but typically it is a small, round object that is used for sitting.', 'A 3D model of a table typically looks like a rectangular object with four legs.', 'A grayscale or white depth map of a tent 3D model would show the tent as a white object against a black background.', 'The depth map of a white toilet would appear as a white object with a black outline.', 'The image is a depth map of a 3D model of a TV stand.', 'A grayscale or white depth map of a vase 3D model would show the contours of the vase in shades of gray or white, with the darkest areas representing the deepest parts of the vase.', 'A grayscale or white depth map of a wardrobe 3D model would show the overall shape and form of the wardrobe, as well as the depth of each component.', 'A grayscale depth map of a 3D Xbox model would show a range of light to dark gray tones, with the darkest areas representing the closest parts of the model to the viewer, and the lightest areas representing the farthest parts of the model.']
best_prompt_weight['modelnet40_vit_b32_test_prompts'] = ['A gray, unclear 3D model of an airplane on an incline.', 'This is a depth map of a bathtub in either grayscale or white.', 'This depth map is either grayscale or white and depicts a bed.', 'An indistinct grayscale depth map of a slanted rough a bench model.', 'This depth map appears to be of a bookshelf, but it is difficult to tell due to the low resolution and poor lighting.', 'If a bottle is shown on a grayscale map, it will appear as a dark object.', 'A 3D model of a bowl composed of gray, obscure balls that is inclined.', 'This is a depth map of a car in either grayscale or white.', 'an unnoticeable, black and white depth map of an off-kilter, coarse a chair model.', 'An inclined cone composed of gray obscure balls, modeled roughly in 3D.', 'A depth map of a cup would look like a picture of a cup with different shades of gray.', 'The obscure grayscale depth map represents an inclined rough curtain which is white.', 'An obscure depth map of a desk 3D model would show the desk in great detail, but the background would be blurred and fuzzy.', 'This is a two-dimensional representation of a door in shades of gray.', 'The dresser is a 3D model of a piece of furniture.', 'A jagged 3D model of an slanted a flower pot composed of murky gray balls.', 'This is either a grayscale or white depth map of a glass box.', 'This grayscale depth map represents a guitar.', 'The keyboard 3D model is a very detailed and accurate representation of a keyboard.', 'If a map is in grayscale, the lamp would be a light gray color.', 'An obscure grayscale depth map of a laptop model that is slanted and has a textured surface.', 'A simple heightmap of a mantel.', 'This is a depth map of a monitor in shades of gray.', 'This depth map is of a night stand and is either grayscale or white.', 'This heightmap shows a person standing in a very simple and obscure environment.', 'This depth map is in grayscale or white and shows a piano.', 'This is a depth map of a plant in shades of gray.', 'This is a depth map of a radio in grayscale.', 'A murky grayscale depth map of a slanted rough a range hood model.', 'a depth map in shades of gray of a slanted, bumpy sink model.', 'An unclear depth map of a slanted couch model.', 'This heightmap represents a stairs.', 'A stool in a shades of gray.', 'A render of a table with four legs, a flat surface, and a slight lip around the edge.', 'This image is a grayscale or white depth map of a tent.', 'This depth map is either shades of gray or white, depicting a toilet.', 'A hazy depth map of a slanted bumpy a TV stand model.', 'A vague 3D model of a vase composed of gray round balls.', 'This is a white depth map of a wardrobe or grayscale.', 'This is a depth map of a 3D model of the Xbox gaming console.']
best_prompt_weight['modelnet40_rn50_test_weights'] = [0.75, 0.75, 0.75, 0.75, 0.25, 0.5, 1.0, 1.0, 0.25, 0.25]
best_prompt_weight['modelnet40_rn101_test_weights'] = [0.75, 0.75, 0.75, 0.25, 0.5, 1.0, 0.75, 0.25, 0.25, 0.25]
best_prompt_weight['modelnet40_vit_b16_test_weights'] = [0.75, 0.75, 0.75, 0.25, 0.75, 1.0, 0.25, 1.0, 0.75, 0.25]
best_prompt_weight['modelnet40_vit_b32_test_weights'] = [0.75, 0.75, 0.75, 0.75, 0.25, 0.5, 0.5, 0.75, 0.5, 0.5]

