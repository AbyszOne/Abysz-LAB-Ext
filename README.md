# Abysz AI temporal coherence lab. 
## Automatic1111 Extension. Alpha 0.0.9b

![Captura de pantalla (79)](https://user-images.githubusercontent.com/112580728/226178563-97d8f1d8-0a29-468e-b4e2-aacaf40cee80.png)

## How DFI works:

https://user-images.githubusercontent.com/112580728/226049549-e61bddb3-88ea-4953-893d-9993dd165180.mp4

# Requirements

OpenCV: ```pip install opencv-python```

Imagemagick library: https://imagemagick.org/script/download.php

## Basic guide:
Differential frame interpolation analyzes the stability of the original video, and processes the generated video with that information. Example, if your original background is static, it will force the generated video to respect that, acting as a complex deflicker. It is an aggressive process, for which we need and will have a lot of control.

Gui version 0.0.6 includes the following parameters.

**Frame refresh frequency:** Every how many frames the interpolation is reduced. It allows to keep more information of the generated video, and avoid major ghosting.

**Refresh Strength:** Opacity % of the interpolated information. 0 refreshes the entire frame, with no changes. Here you control how much change you allow overall.

**DFI Strength:** Amount of information that tries to force. 4-6 recommended.

**DFI Deghost:** A variable that generally reduces the areas affected by DFI. This can reduce ghosting without changing DFI strength.

**Smooth:** Smoothes the interpolation. High values reduce the effectiveness of the process.

**Source denoise:** Improves scanning in noisy sources.

(DEFLICKERS PLAYGROUND ADDED)

# USE STRATEGIES:

### Basic: 
The simplest use is to find the balance between deflicking and deghosting. However, this is not efficient.

## Multipass:
The most efficient way to use this tool is to allow a certain amount of corruption and ghosting, in exchange for more stable video. Once we have that base, we must use a second step in Stable Diffusion, at low denoising (1-4). In most cases, this brings back much of the detail, but retains the stability we've gained.

# Multibatch-controlnet: 
The best, best way to use this tool is to use our "stabilized" video in img2img, and the original (REAL) video in controlnet HED. Then use a parallel batch to retrieve details. This considerably improves the multipass technique. Unfortunately, that function is not available in the controlnet gui as of this writing.

# TODO
Automatic1111 extension. Given my limited knowledge of programming, I had trouble getting my script to interact within A1111. I hope soon to solve details to integrate this tool.
Also, there are many important utilities that are in development, waiting to be added soon, such as polar rendering (like "front/back", but more complex), gif viewer, source analysis, preprocessing, etc.


