from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from ultraflux.pipeline_flux import FluxPipeline
from ultraflux.transformer_flux_visionyarn import FluxTransformer2DModel
from ultraflux.autoencoder_kl import AutoencoderKL
import os

local_vae = AutoencoderKL.from_pretrained("Owen777/UltraFlux-v1",subfolder="vae", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained("Owen777/UltraFlux-v1",subfolder="transformer",torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("Owen777/UltraFlux-v1", vae=local_vae, torch_dtype=torch.bfloat16, transformer=transformer)
from diffusers import FlowMatchEulerDiscreteScheduler
pipe.scheduler.config.use_dynamic_shifting = False
pipe.scheduler.config.time_shift = 4
pipe = pipe.to("cuda")

os.makedirs("results", exist_ok=True) 
prompts = [
    "Photographed in a soft diffused daylight studio setting, a young girl with curly blonde hair sits gracefully against a pale, textured fabric backdrop, the frame capturing a medium close-up with a 50mm lens to emphasize intimacy; she wears a flowing blue and silver patterned shawl draped elegantly around her shoulders, subtly revealing a warm brown textile underneath, her thoughtful expression accompanied by a gentle, knowing smile, with natural makeup and delicate features softly lit from the side, evoking a serene, painterly quality with a warm-neutral palette reminiscent of fine portrait photography.",
    "A lone climber in a vivid red jacket and glowing headlamp traverses a narrow, snow-covered mountain ridge at dusk, framed by dramatic jagged peaks fading into misty clouds. Shot from a mid-distance with a 70mm lens for cinematic depth, the cold blue twilight contrasts with the warm beam of light. Snow swirls softly in the wind, highlighting the climber’s steady determination. The scene glows with moody, diffused side lighting and a cool-toned color palette, resembling a high-resolution cinematic still from an expedition film.",
    "A gray ceramic pitcher with a delicate handle, brimming with soft white flowers, rests on a rustic wooden table. Nearby, three vibrant yellow lemons add a burst of color, their smooth skin contrasting against the textured wood. A single pale flower petal lies gently on the surface, caught in the natural lighting. The scene is bathed in warm, golden sunlight, casting soft shadows and highlighting the earthy tones. The composition evokes a serene, intimate atmosphere, perfect for capturing a timeless, natural still life.",
    "A woman in her late 20s with shoulder-length blonde hair sits comfortably on a vintage armchair in a cozy living room, wearing a soft red cardigan over a white blouse and a vibrant yellow skirt with a subtle floral pattern, deeply focused on the pages of an open book, her expression calm and contemplative, natural light streaming through a window casting warm tones on her face, with a soft bokeh background of a bookshelf and houseplants, the atmosphere cozy and inviting, captured with a medium focal length lens, softly diffused daylight, evoking a serene, peaceful mood.",
    "A pair of golden-brown roasted chickens, their skin crispy and glistening with flavorful seasoning, are elegantly arranged in a rich, aromatic sauce infused with green olives, set on a rustic wooden table. The chickens are accompanied by a vibrant grain salad bursting with a mix of colors, garnished with crunchy nuts, and a delicate small bowl of creamy pale yellow sauce on the side. Soft natural lighting from the left casts gentle shadows across the scene, with warm tones reflecting the cozy, homey atmosphere of a late afternoon kitchen setting. The food is captured with a shallow depth of field to emphasize the textures and rich colors, giving the dish a mouth-watering, inviting appeal.",
    "A middle-aged man with a salt-and-pepper beard and a stylish hat plays an acoustic guitar, dressed in a chic blazer over a dark shirt, standing in a cozy, softly blurred room with a muted, neutral-toned background. The camera captures him from a slight low angle, with a medium close-up shot that emphasizes his focused expression and the elegant movement of his fingers on the strings. Warm, soft key lighting gently highlights his face and guitar, creating a peaceful, intimate atmosphere with a hint of natural sunlight coming from a nearby window, casting a golden hue on the scene.",
    "A breathtaking view of a vibrant night sky filled with a swirling Milky Way galaxy, its cosmic colors reflecting in the serene surface of a still, steaming geothermal pool. The pool is surrounded by dark, shadowy silhouettes of trees, their forms barely visible in the misty atmosphere. Soft, ethereal light glows from the galaxy above, casting gentle reflections on the water, creating a tranquil, otherworldly ambiance. The scene is set during a clear, crisp night, with cool, muted blues and purples dominating the color palette, evoking a sense of peaceful solitude."
    
    "A photorealistic studio portrait of a young woman with olive skin, freckles, and loose chestnut curls, seated before a charcoal muslin backdrop. Soft octagonal lights wrap her face with a creamy gradient, highlighting the fine texture of her silk blouse and the delicate gold rings on her fingers. Gentle lens bloom catches in her dark eyes, conveying calm determination while stray hairs glint against the shallow depth of field.",
    "A cinematic profile shot of an elderly man in a sunlit atelier, his weathered hands resting on a workbench scattered with chisels and wood shavings. Shafts of morning light filter through dusty windows, revealing the texture of his linen shirt and the intricate grain of the half-carved sculpture beside him. The mood is contemplative, with warm amber tones contrasting the cool slate shadows in the corners of the room.",
    "An expansive alpine landscape at sunrise where serrated granite peaks rise above a valley carpeted in lavender wildflowers. Mist curls around the cliffs as glacial streams snake between mossy boulders, reflecting a sky awash in peach and icy blue. The atmosphere feels crisp and serene, with long shadows emphasizing the scale of the untouched wilderness.",
    "A windswept black-sand coastline just after a storm, with turquoise waves exploding against basalt arches and sending silver spray through the air. Wet driftwood litters the beach, and distant mountains fade under bruised clouds tinged with magenta. The lighting is dramatic, with sun rays piercing the gloom and illuminating the swirling seabirds overhead.",
    "A dense rainforest ravine dominated by a tiered waterfall cascading into a jade pool. Giant ferns, luminous orchids, and moss-coated branches frame the scene while humidity hangs as a golden haze. Sunbeams penetrate the canopy, creating glittering droplets that suspend midair and give the entire composition a dreamlike vibrancy.",
    "A desert canyon at blue hour where sculpted sandstone walls glow in gradients of amber, rose, and violet. A narrow river snakes through the gorge, reflecting the first stars as fireflies drift above sparse desert blooms. The stillness is punctuated by a lone traveler leading a camel, their silhouettes etched against the luminous horizon.",
    "A futuristic megacity plaza at midnight, dominated by mirrored skyscrapers whose holographic signage ripples across rain-slick streets. Crowds in reflective ponchos weave between suspended trams, while bioluminescent trees cast aqua light on the polished basalt benches. The scene pulses with neon magentas, electric cyans, and chrome reflections.",
    "An ornate Victorian library interior viewed from a mezzanine, with spiraling iron staircases and towering bookshelves lined with worn leather tomes. Brass lamps and stained-glass skylights bathe the reading tables in honeyed light, dust motes shimmering above scholars immersed in parchment maps. Every surface exudes age and meticulous craftsmanship.",
    "A rustic farmhouse kitchen during golden hour, where a wooden table overflows with heirloom tomatoes, freshly baked sourdough, and sprigs of basil in hand-thrown ceramics. Sunlight pours through gauzy curtains, catching flour particles suspended above a marble counter. The scene radiates warmth, inviting textures, and the promise of a shared meal.",
    "A bustling night market food stall framed by colorful paper lanterns and stainless-steel countertops sizzling with skewers. Vendors fan glowing charcoal while steam rises from bamboo baskets, and handwritten menus dangle from strings. The air is thick with aromas of roasted sesame and chili oil, and the crowd’s motion blurs under long exposure lighting.",
]

for idx, prompt in enumerate(prompts, start=1):
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
