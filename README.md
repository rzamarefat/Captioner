# Captioner
## A bilingual en2end RNN-based approach for captioning images in English and Persian

- ### Overview
![Bilingual_image_captioning (1)](https://user-images.githubusercontent.com/79300456/215791467-aac4fe16-dbdb-46ed-9d42-22ec1fd53217.jpg)

- ### Examples
| Image | English    | Persian    |
| :---:   | :---: | :---: |
| <img src="https://user-images.githubusercontent.com/79300456/215793351-c859c844-b3a6-4971-92c0-464e8124e39e.png" data-canonical-src="https://user-images.githubusercontent.com/79300456/215793351-c859c844-b3a6-4971-92c0-464e8124e39e.png" width="150" height="100" /> | A woman in black shirt is sitting on the floor and playing the guitar   | زنی جوان روی زمین نشسته و در حال نواختن گیتار است |
| <img src="https://user-images.githubusercontent.com/79300456/215793360-24e9b4b5-abc6-4e66-8e76-cbfb2737ce7a.png" data-canonical-src="https://user-images.githubusercontent.com/79300456/215793360-24e9b4b5-abc6-4e66-8e76-cbfb2737ce7a.png" width="150" height="100" /> | A person is riding a brown horse | زنی در حال سوارکاری است   |
| <img src="https://user-images.githubusercontent.com/79300456/215793372-86b68fb0-ce59-4221-8208-8410d8a379fa.png" data-canonical-src="https://user-images.githubusercontent.com/79300456/215793372-86b68fb0-ce59-4221-8208-8410d8a379fa.png" width="150" height="100" /> | Two children are brawling with each other   | دو کودک در حال دعوا کردن هستند   |
| <img src="https://user-images.githubusercontent.com/79300456/215793378-2a90bfcf-bb80-4d5e-a72f-3160cb4c8de4.png" data-canonical-src="https://user-images.githubusercontent.com/79300456/215793378-2a90bfcf-bb80-4d5e-a72f-3160cb4c8de4.png" width="150" height="100" /> | Someone is putting colorful flowers into a vase   | فردی در حال گذاشتن گل های رنگارنگ در داخل یک گلدان شیشه ای است   |

- ### Train
After making a torch env do the following:
1. Download Flicker8k Dataset
2. By cloning the repo you have en_farsi_captions.txt which includes the following: image_id, English caption, Persian caption
2. Set the related path variables in train.py module
- ### Inference
1. Set the related path variables in infer.py module
- ### ToDos
-[] Refactor based on OOP
-[] Provide pretrained ckpts
- ### Refs
- A big :thumbsup: for Aladdin Persson [tutorial](https://www.youtube.com/watch?v=y2BaTt1fxJU)
- I have used [Abadis](https://abadis.ir/) and selenium for automatically translating English ground-truth captions 

