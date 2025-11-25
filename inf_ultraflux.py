from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from ultraflux.pipeline_flux import FluxPipeline
from ultraflux.transformer_flux import FluxTransformer2DModel
from ultraflux.autoencoder_kl import AutoencoderKL
import os

local_vae = AutoencoderKL.from_pretrained("Owen777/UltraFlux-v1",subfolder="vae", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained("Owen777/UltraFlux-v1",subfolder="transformer",torch_dtype=torch.bfloat16)
# transformer = FluxTransformer2DModel.from_pretrained("Owen777/UltraFlux-v1-1-Transformer",torch_dtype=torch.bfloat16) # NOTE:uncomment this line to use UltraFlux-v1.1
pipe = FluxPipeline.from_pretrained("Owen777/UltraFlux-v1", vae=local_vae, torch_dtype=torch.bfloat16, transformer=transformer)
from diffusers import FlowMatchEulerDiscreteScheduler
pipe.scheduler.config.use_dynamic_shifting = False
pipe.scheduler.config.time_shift = 4
pipe = pipe.to("cuda")

os.makedirs("results", exist_ok=True) 
prompts = [    
    "A vast rocky landscape dominated by towering, weathered stone formations, bathed in the ethereal glow of a vibrant night sky filled with a sea of stars, the Milky Way stretching across the heavens, captured from a low angle to emphasize the immense scale of the rocks against the expansive cosmos above. The scene is illuminated by soft, cool moonlight, casting long, dramatic shadows on the textured rock surfaces. The color palette is rich with deep blues, purples, and silvery whites, creating a serene, otherworldly atmosphere.",
    "A breathtaking scene of snow-capped mountains encircling a serene lake, their towering peaks perfectly mirrored in the still water under a twilight sky adorned with soft, colorful clouds. The gentle mist rises from the lake's surface, adding a mystical touch to the tranquil landscape, with the golden hues of the setting sun casting a warm glow over the scene. The composition captures the vastness of the mountains, the calm water, and the ethereal atmosphere, with a focus on soft, natural lighting and a cool color palette that enhances the peaceful mood.",
    "A serene snow-covered countryside unfolds in soft morning light, where a small group of woolly sheep graze in the crisp foreground, their breath visible in the cold air, while in the gently blurred midground, a charming, snow-dusted church with a stone steeple rises among rustic timber houses nestled among tall pine trees, all bathed in a pale blue winter palette under an overcast sky, captured with a 50mm lens for a shallow depth of field and a cinematic, filmic warmth.",
    "A confident woman with short, dark hair stands in a lush forest, her black dress flowing slightly in the breeze. She holds an owl with large, outstretched wings, its feathers sharp and detailed against the dappled sunlight. The light filters softly through the tall trees above, casting intricate shadows on her skin and highlighting her intricate tattoo along her arm. The forest around her feels serene and mystical, with the air thick with the scents of nature. The color palette blends deep greens, earthy browns, and the rich contrast of black and white.",


    "Photographed in a soft diffused daylight studio setting, a young girl with curly blonde hair sits gracefully against a pale, textured fabric backdrop, the frame capturing a medium close-up with a 50mm lens to emphasize intimacy; she wears a flowing blue and silver patterned shawl draped elegantly around her shoulders, subtly revealing a warm brown textile underneath, her thoughtful expression accompanied by a gentle, knowing smile, with natural makeup and delicate features softly lit from the side, evoking a serene, painterly quality with a warm-neutral palette reminiscent of fine portrait photography.",
    "A lone climber in a vivid red jacket and glowing headlamp traverses a narrow, snow-covered mountain ridge at dusk, framed by dramatic jagged peaks fading into misty clouds. Shot from a mid-distance with a 70mm lens for cinematic depth, the cold blue twilight contrasts with the warm beam of light. Snow swirls softly in the wind, highlighting the climber’s steady determination. The scene glows with moody, diffused side lighting and a cool-toned color palette, resembling a high-resolution cinematic still from an expedition film.",
    "A gray ceramic pitcher with a delicate handle, brimming with soft white flowers, rests on a rustic wooden table. Nearby, three vibrant yellow lemons add a burst of color, their smooth skin contrasting against the textured wood. A single pale flower petal lies gently on the surface, caught in the natural lighting. The scene is bathed in warm, golden sunlight, casting soft shadows and highlighting the earthy tones. The composition evokes a serene, intimate atmosphere, perfect for capturing a timeless, natural still life.",
    "A woman in her late 20s with shoulder-length blonde hair sits comfortably on a vintage armchair in a cozy living room, wearing a soft red cardigan over a white blouse and a vibrant yellow skirt with a subtle floral pattern, deeply focused on the pages of an open book, her expression calm and contemplative, natural light streaming through a window casting warm tones on her face, with a soft bokeh background of a bookshelf and houseplants, the atmosphere cozy and inviting, captured with a medium focal length lens, softly diffused daylight, evoking a serene, peaceful mood.",
    "A pair of golden-brown roasted chickens, their skin crispy and glistening with flavorful seasoning, are elegantly arranged in a rich, aromatic sauce infused with green olives, set on a rustic wooden table. The chickens are accompanied by a vibrant grain salad bursting with a mix of colors, garnished with crunchy nuts, and a delicate small bowl of creamy pale yellow sauce on the side. Soft natural lighting from the left casts gentle shadows across the scene, with warm tones reflecting the cozy, homey atmosphere of a late afternoon kitchen setting. The food is captured with a shallow depth of field to emphasize the textures and rich colors, giving the dish a mouth-watering, inviting appeal.",
    "A middle-aged man with a salt-and-pepper beard and a stylish hat plays an acoustic guitar, dressed in a chic blazer over a dark shirt, standing in a cozy, softly blurred room with a muted, neutral-toned background. The camera captures him from a slight low angle, with a medium close-up shot that emphasizes his focused expression and the elegant movement of his fingers on the strings. Warm, soft key lighting gently highlights his face and guitar, creating a peaceful, intimate atmosphere with a hint of natural sunlight coming from a nearby window, casting a golden hue on the scene.",
    "A breathtaking view of a vibrant night sky filled with a swirling Milky Way galaxy, its cosmic colors reflecting in the serene surface of a still, steaming geothermal pool. The pool is surrounded by dark, shadowy silhouettes of trees, their forms barely visible in the misty atmosphere. Soft, ethereal light glows from the galaxy above, casting gentle reflections on the water, creating a tranquil, otherworldly ambiance. The scene is set during a clear, crisp night, with cool, muted blues and purples dominating the color palette, evoking a sense of peaceful solitude.",

    "A cinematic profile shot of an elderly man in a sunlit atelier, his weathered hands resting on a workbench scattered with chisels and wood shavings. Shafts of morning light filter through dusty windows, revealing the texture of his linen shirt and the intricate grain of the half-carved sculpture beside him. The mood is contemplative, with warm amber tones contrasting the cool slate shadows in the corners of the room.",
    "An expansive alpine landscape at sunrise where serrated granite peaks rise above a valley carpeted in lavender wildflowers. Mist curls around the cliffs as glacial streams snake between mossy boulders, reflecting a sky awash in peach and icy blue. The atmosphere feels crisp and serene, with long shadows emphasizing the scale of the untouched wilderness.",
    "A dense rainforest ravine dominated by a tiered waterfall cascading into a jade pool. Giant ferns, luminous orchids, and moss-coated branches frame the scene while humidity hangs as a golden haze. Sunbeams penetrate the canopy, creating glittering droplets that suspend midair and give the entire composition a dreamlike vibrancy.",
    "A rustic farmhouse kitchen during golden hour, where a wooden table overflows with heirloom tomatoes, freshly baked sourdough, and sprigs of basil in hand-thrown ceramics. Sunlight pours through gauzy curtains, catching flour particles suspended above a marble counter. The scene radiates warmth, inviting textures, and the promise of a shared meal.",
    "A bustling night market food stall framed by colorful paper lanterns and stainless-steel countertops sizzling with skewers. Vendors fan glowing charcoal while steam rises from bamboo baskets, and handwritten menus dangle from strings. The air is thick with aromas of roasted sesame and chili oil, and the crowd’s motion blurs under long exposure lighting.",
    "A top-down gourmet plating of matcha mille crepe cake beside crystalline yuzu jellies on a matte slate board. Dewy berries glisten under controlled studio light, and powdered sugar drifts like snow onto the sculpted dessert layers. Reflections are carefully managed to showcase the velvety textures and vibrant palette.",
    "A serene Japanese tea ceremony set on tatami mats, with a host in indigo kimono whisking emerald tea in a raku bowl. Paper shoji screens diffuse soft daylight, accenting the minimal ikebana arrangement and lacquered utensils. The mood is meditative, every element emphasizing intentional craftsmanship.",
    "An industrial sci-fi hangar housing a sleek explorer spacecraft suspended by maglev cranes, floodlit by cool white panels. Engineers in exosuits hover on maintenance platforms, sparks cascading as they weld carbon fiber plating. Vapor trails drift across the polished floor, capturing reflections of neon status displays.",
    "A high-fantasy citadel perched on a cliff under a moonlit sky, its spires adorned with stained glass and banners fluttering in the night wind. Bioluminescent vines climb the stone walls while wyverns circle above, their scales catching the silver glow. Torches along the bridge cast rhythmic shadows, creating a sense of impending adventure.",
    "A polar night landscape where auroras unfurl above an icy fjord, reflecting emerald and violet ribbons onto the mirrored water. Snow-dusted pines frame a wooden cabin emitting a faint amber glow, and the distant mountains glow with moonlit edges. The stillness is profound, with only faint ice fog drifting across the foreground.",
    "An anime-inspired magical girl poised on a floating crystal platform, her pastel hair billowing as constellations swirl behind her. Intricate armor accented with glowing runes refracts prismatic light, and her staff releases streams of glittering particles. The scene blends dynamic action lines with painterly gradients for dramatic flair.",
    "A dynamic anime mecha battle staged above a futuristic wasteland, with two towering robots trading energy blades that ignite the smoky dusk sky. Shockwaves ripple through shattered skyscrapers, and debris glows from residual plasma. Bold cel-shaded highlights emphasize their articulated armor and glowing visors.",
    "A cozy chibi-style cafe interior populated by round-faced characters sipping oversized lattes beneath string lights. Wooden beams, chalkboard menus, and potted succulents create a charming atmosphere, while pastries shaped like tiny animals fill glass displays. Soft pastel tones and gentle gradients evoke playful warmth.",
    "A retro-futuristic diner bathed in neon magenta and teal, with chrome booths, reflective checkered floors, and a row of milkshakes topped with whipped cream. Patrons in holographic jackets chat with a friendly android barista, jukebox lights pulsing to synthwave beats. The scene combines nostalgic details with sleek sci-fi embellishments."

]




from pathlib import Path

for idx, prompt in enumerate(prompts, start=1):
    out_path = Path("results") / f"ultra_flux_{idx:02d}.jpeg"
    image = pipe(
        prompt,
        height=4096,
        width=4096,
        guidance_scale=4,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    image.save(f"results/ultra_flux_{idx:02d}.jpeg")
