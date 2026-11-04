---
license: cc-by-nc-sa-4.0
task_categories:
  - image-classification
language:
  - en
tags:
  - Disaster
  - Crisis Informatics
pretty_name: 'CrisisMMD: Multimodal Twitter Datasets from Natural Disasters'
size_categories:
  - 10K<n<100K
dataset_info:
- config_name: humanitarian
  splits:
    - name: train
      num_examples: 13608
    - name: dev
      num_examples: 2237
    - name: test
      num_examples: 2237
  features:
    - name: event_name
      dtype: string
      description: "Name of the disaster event, such as 'hurricane_maria'."
    - name: tweet_id
      dtype: string
      description: "Unique identifier for the tweet."
    - name: image_id
      dtype: string
      description: "Unique identifier for the image associated with the tweet."
    - name: tweet_text
      dtype: string
      description: "The text content of the tweet."
    - name: image_path
      dtype: string
      description: "File path to the image."
    - name: image
      dtype: Image
      description: "Image data loaded directly from file."
    - name: label
      dtype:
        class_label:
          names:
            '0': affected_individuals
            '1': infrastructure_and_utility_damage
            '2': injured_or_dead_people
            '3': missing_or_found_people
            '4': not_humanitarian
            '5': other_relevant_information
            '6': rescue_volunteering_or_donation_effort
            '7': vehicle_damage
      description: "Humanitarian classification label for the tweet."
- config_name: informative
  splits:
    - name: train
      num_examples: 13608
    - name: dev
      num_examples: 2237
    - name: test
      num_examples: 2237
  features:
    - name: event_name
      dtype: string
      description: "Name of the disaster event, such as 'hurricane_maria'."
    - name: tweet_id
      dtype: string
      description: "Unique identifier for the tweet."
    - name: image_id
      dtype: string
      description: "Unique identifier for the image associated with the tweet."
    - name: tweet_text
      dtype: string
      description: "The text content of the tweet."
    - name: image_path
      dtype: string
      description: "File path to the image."
    - name: image
      dtype: Image
      description: "Image data loaded directly from file."
    - name: label
      dtype:
        class_label:
          names:
            '0': informative
            '1': not_informative
      description: "Informativeness classification label for the tweet."
- config_name: damage
  splits:
    - name: train
      num_examples: 2468
    - name: dev
      num_examples: 529
    - name: test
      num_examples: 529
  features:
    - name: event_name
      dtype: string
      description: "Name of the disaster event, such as 'hurricane_maria'."
    - name: tweet_id
      dtype: string
      description: "Unique identifier for the tweet."
    - name: image_id
      dtype: string
      description: "Unique identifier for the image associated with the tweet."
    - name: tweet_text
      dtype: string
      description: "The text content of the tweet."
    - name: image_path
      dtype: string
      description: "File path to the image."
    - name: image
      dtype: Image
      description: "Image data loaded directly from file."
    - name: label
      dtype:
        class_label:
          names:
            '0': little_or_no_damage
            '1': mild_damage
            '2': severe_damage
      description: "Damage severity classification label for the tweet."
configs:
- config_name: humanitarian
  data_files:
    - split: train
      path: humanitarian/train.json
    - split: dev
      path: humanitarian/dev.json
    - split: test
      path: humanitarian/test.json
- config_name: informative
  data_files:
    - split: train
      path: informative/train.json
    - split: dev
      path: informative/dev.json
    - split: test
      path: informative/test.json
- config_name: damage
  data_files:
    - split: train
      path: damage/train.json
    - split: dev
      path: damage/dev.json
    - split: test
      path: damage/test.json
---

# CrisisMMD: Multimodal Twitter Datasets from Natural Disasters

The **CrisisMMD** multimodal Twitter dataset consists of several thousand manually annotated tweets and images collected during seven major natural disasters, including earthquakes, hurricanes, wildfires, and floods from 2017. The dataset includes three types of annotations:

On HuggingFace, we hosted version 2.0 of the CrisisMMD dataset. Please see further information below.

### Disaster Response Tasks

1. **Task 1: Informative vs Not Informative**
   - Informative
   - Not informative
   - "Don't know or can't judge" → **Removed in version 2.0**

2. **Task 2: Humanitarian Categories**
   - Affected individuals
   - Infrastructure and utility damage
   - Injured or dead people
   - Missing or found people
   - Rescue, volunteering, or donation effort
   - Vehicle damage
   - Other relevant information
   - "Not relevant or can't judge" → **Updated to "Not humanitarian" in version 2.0**

3. **Task 3: Damage Severity Assessment**
   - Severe damage
   - Mild damage
   - Little or no damage
   - "Don't know or can't judge"

## Datasets Details

The keywords used for collecting tweets, along with the start and end dates for each event, are outlined in the following table.

| Crisis Name        | Keywords                                       | Start Date        | End Date          |
|--------------------|------------------------------------------------|-------------------|-------------------|
| [Hurricane Irma](https://en.wikipedia.org/wiki/Hurricane_Irma)      | Hurricane Irma, Irma storm, Storm Irma, etc. | Sep 6, 2017       | Sep 21, 2017      |
| [Hurricane Harvey](https://en.wikipedia.org/wiki/Hurricane_Harvey)  | Hurricane Harvey, Tornado, etc.              | August 25, 2017   | September 20, 2017|
| [Hurricane Maria](https://en.wikipedia.org/wiki/Hurricane_Maria)    | Hurricane Maria, Maria Storm, etc.           | September 20, 2017| November 13, 2017 |
| [California wildfires](https://en.wikipedia.org/wiki/List_of_California_wildfires) | California fire, USA Wildfire, etc. | October 10, 2017   | October 27, 2017   |

### Event-wise data distribution

For each event, we collected tweets and associated images, filtered and sampled them for the annotation.

## [**Data distribution from the CrisisMMD version v1.0**](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v1.0.tar.gz)

| Crisis Name            | # Tweets    | # Images   | # Filtered Tweets | # Sampled Tweets | # Sampled Images |
|------------------------|-------------|------------|-------------------|------------------|------------------|
| Hurricane Irma         | 3,517,280   | 176,972    | 5,739            | 4,041           | 4,525           |
| Hurricane Harvey       | 6,664,349   | 321,435    | 19,967           | 4,000           | 4,443           |
| Hurricane Maria        | 2,953,322   | 52,231     | 6,597            | 4,000           | 4,562           |
| California wildfires   | 455,311     | 10,130     | 1,488            | 1,486           | 1,589           |
| Mexico earthquake      | 383,341     | 7,111      | 1,241            | 1,239           | 1,382           |
| Iraq-Iran earthquake   | 207,729     | 6,307      | 501              | 499             | 600             |
| Sri Lanka floods       | 41,809      | 2,108      | 870              | 832             | 1,025           |
| **Total**              | **14,223,141** | **576,294** | **36,403**       | **16,097**      | **18,126**      |

## Data preparation for multimodal baseline
For the multimodal baseline experiments, we first combined the tweet text and image from all events. It resulted in 24 duplicate entries (tweet ids: text and associated images). We manually checked these duplicate entries and kept the one, which were annotated properly. We changed the label “Not relevant or can’t judge” to “Not humanitarian”. In addition, as the annotation consists of a label - “don't know or can't not judge”, we also removed them for the classification experiments. Hence, this preprocessing part filtered out 39 tweets and associated 44 images. The resulted total dataset consists of 16058 and 18082 tweet texts and images, respectively as shown in the following table. This version of this dataset is released as version 2.0 and is available for download.

## [**Data distribution from the CrisisMMD version v2.0**](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)

In this version, the "Not relevant or can't judge" label has been mapped to "Not humanitarian" for the humanitarian task. Additionally, the "Not informative" label from the informative task has also been mapped to "Not humanitarian" for the humanitarian task. Duplicate entries from different events have been removed.

### Informativeness

|               | Text   | Image  |
|---------------|--------|--------|
| Informative   | 11,509 | 9,374  |
| Not informative | 4,549 | 8,708  |
| **Total**     | 16,058 | 18,082 |

### Humanitarian

|                               | Text   | Image |
|-------------------------------|--------|-------|
| Affected individuals          | 472    | 562   |
| Infrastructure and utility damage | 1,210 | 3,624 |
| Injured or dead people        | 486    | 110   |
| Missing or found people       | 40     | 14    |
| Not humanitarian              | 4,549  | 8,708 |
| Other relevant information    | 5,954  | 2,529 |
| Rescue, volunteering, or donation effort | 3,293 | 2,231 |
| Vehicle damage                | 54     | 304   |
| **Total**                     | 16,058 | 18,082 |

### Damage Severity

|                 | Text | Image |
|-----------------|------|-------|
| Little or no damage | -    | 475   |
| Mild damage         | -    | 839   |
| Severe damage       | -    | 2,212 |
| **Total**           | -    | 3,526 |


## Downloads (Alternate options)

- **CrisisMMD dataset version v2.0**: [Download labeled images and tweets (~1.8GB)](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)
- **Datasplit**: [Annotations Download](https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_all.zip)
- **Datasplit for multimodal baseline with agreed labels**: [Annotations Download](https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip)

## Citation
**Please cite the following papers if you use any of these resources in your research.**

1. [Ferda Ofli](https://sites.google.com/site/ferdaofli/), [Firoj Alam](https://firojalam.one/), and [Muhammad Imran](http://mimran.me/), [**Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response**](https://arxiv.org/abs/2004.11838), In Proceedings of the 17th International Conference on Information Systems for Crisis Response and Management (ISCRAM), 2020, USA.

2. [Firoj Alam](https://firojalam.one/), [Ferda Ofli](https://sites.google.com/site/ferdaofli/), and [Muhammad Imran](http://mimran.me/), [**CrisisMMD: Multimodal Twitter Datasets from Natural Disasters**](https://arxiv.org/pdf/1805.00713.pdf), In Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM), 2018, Stanford, California, USA.

```
@InProceedings{crisismmd2018icwsm,
  author = {Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  title = {{CrisisMMD}: Multimodal Twitter Datasets from Natural Disasters},
  booktitle = {Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)},
  year = {2018},
  month = {June},
  date = {23-28},
  location = {USA}
}
@inproceedings{multimodalbaseline2020,
Author = {Ferda Ofli and Firoj Alam and Muhammad Imran},
Booktitle = {17th International Conference on Information Systems for Crisis Response and Management},
Keywords = {Multimodal deep learning, Multimedia content, Natural disasters, Crisis Computing, Social media},
Month = {May},
Organization = {ISCRAM},
Publisher = {ISCRAM},
Title = {Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response},
Year = {2020}
}

```
