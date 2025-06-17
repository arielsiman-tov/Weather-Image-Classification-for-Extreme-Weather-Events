# Weather_Image_Classification_for_Extreme_Weather_Events

### We encourage you to take a look also at the attached presentation for further information.

![image](https://github.com/user-attachments/assets/1c72bd82-21c8-4d93-b8e6-893a4b5a496c)

## Our Project Goal:
**Build a reliable and interpretable model that classifies extreme weather from images – enhancing early detection, situational awareness, and decision-making.**

## What Makes Our Project Unique: 
We go beyond standard image classifiers like R-CNN by enhancing accuracy through multimodal learning :
1. **Expanded Dataset** - Combined real and synthetic images (via diffusion model) for broader coverage of weather and disaster scenarios.
2. **Embedding Validation** – Verified synthetic image quality using t-SNE and cosine- similarity to ensure accurate class alignment.
3. **Multimodal Learning** - Integrated visual features, CLIP-generated textual descriptions, and numerical data for richer, more robust classification.

## Our Workflow Pipeline: 

![image](https://github.com/user-attachments/assets/3ca35781-0724-4ee5-8fcf-00734a722fa0)

## Data Sources:
**1.Kaggle Weather Image Classification Dataset:**   
- Link: Kaggle Dataset(1) , Kaggle Dataset(2)
- Description: Contains labeled images across weather categories: lightning, hail, sandstorm, rain, cyclone (tornado), wildfire and snow.
- Structure: The dataset is organized into train/ and test/ directories, with each subdirectory named after the weather condition it contains.
- Usage: Serves as the primary real-world dataset for initial training, evaluation, and understanding of natural versus non-natural disaster patterns.
  	
**2.Synthetic Images Generated with Stable Diffusion:**  
- Tools:
  * Pipe1: StableDiffusionPipeline from HuggingFace’s diffusers library (runwayml/stable-diffusion-v1-5) which gets prompt as input and generates a picture.
  * Pipe2: StableDiffusionImg2ImgPipeline from HuggingFace’s for controlled image variation, which takes a prompt and an example picture and generates new pictures.
- Description: Contains images across various weather categories: tornado, clouds, tsunami, waves, fire and wildfire.
- Purpose: Expand the dataset to include a broader range of natural phenomena, especially extreme weather scenarios and their visual manifestations.

## Describe Data
- Format: Each image is in standard .png/.jpg/.jpeg format.
- Structure: Organized into main directory – “Natural_Phenomena_Dataset”, inside there is a folder to each weather category (11 categories in total).
- All generated images include the phrase “gen” in their names.
- Missing values and blank fields: No missing values were found in the dataset.
- Quantity:  Each category has between 200-700 images – total 4840 images.
  * Train ~4000 images
  * Test ~1000 images
  * Validation ~1000 images
   
**1. Class Distribution**
![image](https://github.com/user-attachments/assets/169371ea-8651-4cdf-a372-21afbd4aff32)

We aimed to achieve a balanced distribution of images across individual classes as well as between the broader categories of natural disasters and non-natural disasters. Additionally, we ensured diversity among images within the same category to enhance variability and reduce redundancy.
Even in cases where there is some imbalance in the number of images per class, each category is still represented by a sufficient number of samples to ensure a realistic and meaningful depiction.
To address class imbalance, we enriched the dataset during the image generation process by adding representative samples to newly introduced or underrepresented categories (add representative images to new categories or categories with relatively few images).

**2. Examples of images from each class**

![image](https://github.com/user-attachments/assets/6e92fa7f-5263-4a95-88e1-ab0abc31e72a)

## Exploratory Data Analysis (EDA)

**Histogram Observations by Class:**
![image](https://github.com/user-attachments/assets/6d136967-2209-4abb-b289-747507cfbd76)


<img width="494" alt="image" src="https://github.com/user-attachments/assets/0961eee1-f9a5-49e6-9084-1be201cc0aba" />


**Initial Hypotheses & Impact**
1)	Brightness Distribution as a Feature: Many weather types show distinct intensity patterns (e.g., Snow and Hail are bright, Rain and Fire are dark). Including global intensity features could improve classification performance.
2)	Bimodality and Texture: Classes like Hail and Sandstorm show bimodal/spiky behavior, suggesting underlying structural patterns.

**Plot comparison groups:**

The six histogram comparisons were carefully selected to contrast natural phenomena that are either visually and physically similar or represent normal versus extreme manifestations of the same environment.

![image](https://github.com/user-attachments/assets/0e2213d8-da9e-40c6-8c9d-65acd08dfb8d)

Through pixel intensity analysis, clear patterns phenomena like snow, clouds, and hail exhibit high brightness due to their reflective surfaces or bright skies, whereas events such as tornadoes, wildfires, and sandstorms show darker intensity profiles, consistent with low-visibility, turbulent conditions.

In cases like waves vs tsunami or fire vs wildfire, both happen in similar environments (like the sea or areas with fire), but the disasters (tsunami and wildfire) show wider and darker pixel patterns. 
This shows that they are more intense and cause more damage. In the case of lightning and tornadoes, even though both come from storms, lightning has short, bright flashes, while tornadoes look much darker. 

Overall, the histograms help show the difference between normal weather and dangerous natural disasters. These patterns can help us tell similar weather events apart based on how their images look.

## Embedding Validation

**1)	t-SNE clustering Analysis:**

![image](https://github.com/user-attachments/assets/469f0535-606c-4d78-bd45-3607e4d5e60e)

**Intra-class Similarity** - Strong intra-class similarity is evident across most weather classes, with points forming cohesive clusters, particularly strong cohesion in: Lightning (red cluster), Fire (orange cluster), Waves (yellow cluster) -and Tornado (dark pink) - tightly grouped with minimal dispersion.
This suggests images within these classes consistently capture distinctive visual characteristics of their respective weather phenomena.

**Intra-class Diversity** - Classes exhibit healthy internal variation rather than appearing as single points, indicating diversity, in particular: Hail (green), Wildfire (light blue) and Snow (pink) shows significant spread, suggesting diverse visual representations. 

**Inter-class Separability** - Most classes form distinct, well-separated clusters in the embedding space. In particularly clear boundaries exist between: 
- Fire (orange) and waves (yellow)
- Lightning (red) and tsunami (dark blue). 

On the other hand, there are some concerning proximity between: 
- Hail (green) and some lightning (red) instances
- Rain (purple) and sandstorm (brown) in some regions
- Some overlap between different storm-related classes

**2)	Cosine Similarity Matrix Analysis:**

![image](https://github.com/user-attachments/assets/0ce9ff7d-7b29-48aa-8604-adeb0e2b69eb)

**Intra-class Similarity** - Perfect self-similarity (1.00) along the diagonal, confirming consistent representation within classes. All classes have internal coherence in their embedding space.

**Inter-class Relationships:**
Highest similarity pairs (potential confusion areas): 
* Tornado and lightning (0.88) - both involve electrical storm activity
* Clouds and tornado (0.86) - tornadoes often form amid specific cloud formations
* Clouds and waves (0.86) - may share visual patterns or color distributions
* Rain and sandstorm (0.86) - both involve particulate matter in air
* Rain and snow (0.86) - related precipitation phenomena
  
Most distinctive classes (lowest similarity to others):
* Fire and rain (0.66) - very distinct visual properties
* Fire and tsunami (0.68) - minimal shared visual characteristics
* Wildfire and waves (0.70) - opposite elemental phenomena (fire vs. water)
  
Moderately similar classes (0.75-0.85 range): Most weather classes show moderate similarity, reflecting shared visual elements (sky backgrounds, atmospheric conditions)

**Class Distinctiveness Analysis:**
- Fire is the most distinctive class, with generally lower similarity to other weather types (average similarity ~0.73)
- Clouds show high similarity to several classes (particularly waves and tornado), suggesting they may appear as elements in other weather phenomena.
- The similarity matrix reveals that while classes are distinct, they exist in a continuum of visual relationships rather than as completely isolated categories.

Our analysis of CLIP embeddings through t-SNE visualization and cosine similarity metrics provides valuable insights into the weather image dataset quality. The data demonstrates strong intra-class similarity with appropriate diversity, suggesting it effectively captures the visual essence of each weather phenomenon. While most classes show good separability, the natural visual relationships between certain weather phenomena (like tornado-lightning and rain-snow) present classification challenges that should be addressed through targeted dataset improvements.

We assume that certain natural phenomena are likely to share common characteristics and believe that this is natural and correct for the sake of challenging the classification model tasks. 

Our ways of dealing with problems with data diversity in a particular class were:
- Use the diffusion model to generate a wider range of situations that describe the natural phenomenon (different angles of photography - satellite photography, ground photography, aerial photography) as well as to include different backgrounds (road, city, field, metropolis, sunset, sunrise, clear or stormy weather) or to ensure that the image contains accompanying objects such as people, vehicles, houses and other objects in the context.
- Increase the number of images in the database by creating a larger number of images.
- Adding more classes (our starting point was only 5 weather categories).

## Create Descriptions Embeddings Via CLIP

**How CLIP Works for Image Description **

**Step 1: Contrastive Pre-training** 
- CLIP was trained on millions of image-text pairs
- The Image Encoder (green) processes images into embeddings (I₁, I₂, I₃, etc.)
- The Text Encoder (purple) processes captions into embeddings (T₁, T₂, T₃, etc.)
- During training, CLIP learns to make corresponding image-text pairs have similar embeddings while making non-corresponding pairs dissimilar
  
**Step 2: Creating the Weather Description Classifier**
- You provide text descriptions of weather phenomena: "sunny", "rainy", "cloudy", "snowy", etc.
- The Text Encoder converts each weather description into text embeddings
- This creates a "classifier" where each weather type has a corresponding text embedding
  
**Step 3: Zero-shot Prediction**
- Your weather image goes through the Image Encoder to get an image embedding
- CLIP compares this image embedding to all the pre-computed weather description embeddings
- The weather description with the highest similarity score becomes the prediction

## Image Numeric Features Engeneering

**brightness / contrast**
- Calculated from pixel values, these features capture the overall lightness and tonal variation of the image
- They are often indicative of specific weather phenomena (e.g., dark storms vs. bright snow)

**avg_r, avg_g, avg_b**
-  These values represent the average intensity of red, green, and blue channels, respectively
-  They provide a summary of the image’s dominant color palette.

**edge_density**
-  This feature quantifies the amount of edge information in an image, computed using edge detection techniques
-  It reflects texture and structure, which can help distinguish between smooth phenomena (like clouds) and chaotic ones (like wildfires or tornadoes)

## Our Multimodal Classifier Architecture

![image](https://github.com/user-attachments/assets/ea080fc6-e1f3-48ee-ab3b-a5a4b461bec0)

For the task of **binary classification of weather-related disasters**, we selected a **multimodal neural network classifier** that integrates visual, textual, and numerical data sources. The core modeling technique is based on deep learning with modality-specific encoders followed by a joint classification head. The following components were used:
- Visual Modality: Precomputed CLIP image embeddings (ViT-B/32) represent the image input.
- Textual Modality: CLIP-encoded textual descriptions, tokenized and normalized using the CLIP tokenizer.
- Numerical Modality: Hand-crafted numerical features (e.g., brightness, contrast, RGB averages, edge density).

Each modality is passed through a dedicated fully connected (FC) layer to project all feature types into a common latent dimension. These are then concatenated and passed through a final classifier composed of linear, ReLU, dropout, and softmax layers.

The model was trained using the Adam optimizer, with cross-entropy loss for binary classification. This architecture enables flexible evaluation using different combinations of modalities (unimodal, bimodal, and full multimodal).


## Evaluation

![image](https://github.com/user-attachments/assets/5e7b7d6f-2a3b-4bc4-b120-262cbf2b2218)

![image](https://github.com/user-attachments/assets/0b7792f6-de28-41fc-8ebf-124db49b4c61)

## Overfitting Analysis

![image](https://github.com/user-attachments/assets/89cef87b-7896-44c0-9e8b-fbecd3d7bb2d)

![image](https://github.com/user-attachments/assets/1c8d8f10-c55f-468b-80c8-ccce811f9fbc)


## Key Findings
- **Perfect Visual Performance:** Image modality achieves perfect classification (1.0 across all metrics)
- **Visual Dominance:** Any combination including images shows exceptional performance (≥0.996 F1)
- **Multimodal Robustness:** Full multimodal approach achieves near-perfect results (0.996 F1)
- **Optimal Bimodal Combination:** Image + Text combination maintains perfect performance
- **Numerical Limitations:** Hand-crafted numerical features show weakest individual performance

## Future Work
- **Advanced Feature Fusion:** Implement cross-modal attention mechanisms to better integrate visual, textual, and numerical information
- **Transformer-based Architecture:** Replace FC layers with transformer blocks for improved feature interactions
- **Learned Numerical Features:** Replace hand-crafted features with deep convolutional feature extractors
- **Robustness:** Enhancing the model's robustness by incorporating a wider range of weather conditions, while introducing noisy images to increase complexity and improve the effectiveness of the training process









