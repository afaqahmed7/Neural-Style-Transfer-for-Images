# Neural-Style-Transfer-for-Images
Besides Deep Dream, another major development in deep learning-driven image modification that happened in the summer of 2015 is neural style transfer, introduced by Leon Gatys et al. The neural style transfer algorithm has undergone many refinements and spawned many variations since its original introduction, including a viral smartphone app, called Prisma. For simplicity, this section focuses on the formulation described in the original paper.
Neural style transfer consists in applying the "style" of a reference image to a target image, while conserving the "content" of the target image:
 ![alt text](https://s3.amazonaws.com/book.keras.io/img/ch8/style_transfer.png)
 
What is meant by "style" is essentially textures, colors, and visual patterns in the image, at various spatial scales, while the "content" is the higher-level macrostructure of the image. For instance, blue-and-yellow circular brush strokes are considered to be the "style" in the above example using Starry Night by Van Gogh, while the buildings in the Tuebingen photograph are considered to be the "content".

The idea of style transfer, tightly related to that of texture generation, has had a long history in the image processing community prior to the development of neural style transfer in 2015. However, as it turned out, the deep learning-based implementations of style transfer offered results unparalleled by what could be previously achieved with classical computer vision techniques, and triggered an amazing renaissance in creative applications of computer vision.

The key notion behind implementing style transfer is same idea that is central to all deep learning algorithms: we define a loss function to specify what we want to achieve, and we minimize this loss. We know what we want to achieve: conserve the "content" of the original image, while adopting the "style" of the reference image. If we were able to mathematically define content and style, then an appropriate loss function to minimize would be the following:

loss = distance(style(reference_image) - style(generated_image)) +
       distance(content(original_image) - content(generated_image))

Where distance is a norm function such as the L2 norm, content is a function that takes an image and computes a representation of its "content", and style is a function that takes an image and computes a representation of its "style".
Minimizing this loss would cause style(generated_image) to be close to style(reference_image), while content(generated_image) would be close to content(generated_image), thus achieving style transfer as we defined it.
A fundamental observation made by Gatys et al is that deep convolutional neural networks offer precisely a way to mathematically defined the style and content functions. Let's see how.

### The content loss
As you already know, activations from earlier layers in a network contain local information about the image, while activations from higher layers contain increasingly global and abstract information. Formulated in a different way, the activations of the different layers of a convnet provide a decomposition of the contents of an image over different spatial scales. Therefore we expect the "content" of an image, which is more global and more abstract, to be captured by the representations of a top layer of a convnet.
A good candidate for a content loss would thus be to consider a pre-trained convnet, and define as our loss the L2 norm between the activations of a top layer computed over the target image and the activations of the same layer computed over the generated image. This would guarantee that, as seen from the top layer of the convnet, the generated image will "look similar" to the original target image. Assuming that what the top layers of a convnet see is really the "content" of their input images, then this does work as a way to preserve image content.

### The style loss
While the content loss only leverages a single higher-up layer, the style loss as defined in the Gatys et al. paper leverages multiple layers of a convnet: we aim at capturing the appearance of the style reference image at all spatial scales extracted by the convnet, not just any single scale.
For the style loss, the Gatys et al. paper leverages the "Gram matrix" of a layer's activations, i.e. the inner product between the feature maps of a given layer. This inner product can be understood as representing a map of the correlations between the features of a layer. These feature correlations capture the statistics of the patterns of a particular spatial scale, which empirically corresponds to the appearance of the textures found at this scale.
Hence the style loss aims at preserving similar internal correlations within the activations of different layers, across the style reference image and the generated image. In turn, this guarantees that the textures found at different spatial scales will look similar across the style reference image and the generated image.

### In short
In short, we can use a pre-trained convnet to define a loss that will:
Preserve content by maintaining similar high-level layer activations between the target content image and the generated image. The convnet should "see" both the target image and the generated image as "containing the same things".
Preserve style by maintaining similar correlations within activations for both low-level layers and high-level layers. Indeed, feature correlations capture _textures_: the generated and the style reference image should share the same textures at different spatial scales.

Now let's take a look at a Keras implementation of the original 2015 neural style transfer algorithm. As you will see, it shares a lot of similarities with the Deep Dream implementation we developed in the previous section.

### Neural style transfer in Keras
Neural style transfer can be implemented using any pre-trained convnet. Here we will use the VGG19 network, used by Gatys et al in their paper. VGG19 is a simple variant of the VGG16 network we introduced in Chapter 5, with three more convolutional layers.

This is our general process:
Set up a network that will compute VGG19 layer activations for the style reference image, the target image, and the generated image at the same time.

Use the layer activations computed over these three images to define the loss function described above, which we will minimize in order to achieve style transfer.
Set up a gradient descent process to minimize this loss function.

Keep in mind that what this technique achieves is merely a form of image re-texturing, or texture transfer. It will work best with style reference images that are strongly textured and highly self-similar, and with content targets that don't require high levels of details in order to be recognizable. It would typically not be able to achieve fairly abstract feats such as "transferring the style of one portrait to another". The algorithm is closer to classical signal processing than to AI, so don't expect it to work like magic!

Additionally, do note that running this style transfer algorithm is quite slow. However, the transformation operated by our setup is simple enough that it can be learned by a small, fast feedforward convnet as well -- as long as you have appropriate training data available. Fast style transfer can thus be achieved by first spending a lot of compute cycles to generate input-output training examples for a fixed style reference image, using the above method, and then training a simple convnet to learn this style-specific transformation. Once that is done, stylizing a given image is instantaneous: it's a just a forward pass of this small convnet.

### Take aways
Style transfer consists in creating a new image that preserves the "contents" of a target image while also capturing the "style" of a reference image.
"Content" can be captured by the high-level activations of a convnet.
"Style" can be captured by the internal correlations of the activations of different layers of a convnet.
Hence deep learning allows style transfer to be formulated as an optimization process using a loss defined with a pre-trained convnet.
Starting from this basic idea, many variants and refinements are possible!

