#  Instruction-Driven Fusion of Infrared-Visible Images: Tailoring for Diverse Downstream Tasks

The code of "Instruction-Driven Fusion of Infrared-Visible Images: Tailoring for Diverse Downstream Tasks"

## Recommended Environment<br>
The recommended environment to run the code:
 - [ ] python = 3.9.0
 - [ ] torch = 2.3.0
 - [ ] torchvision = 0.18.0
 - [ ] cuda = 11.8
 - [ ] timm = 0.9.12
 - [ ] numpy = 1.24.3
 - [ ] scipy = 1.13.1
 - [ ] pillow = 10.3.0
 - [ ] tensorboardX = 2.6.2.2
 - [ ] opencv-python = 4.9.0.80
 - [ ] mmcv = 2.2.0
 - [ ] kornia = 0.5.11
 

## Getting Started
### To Test:

    torchrun --nproc_per_node 1 test_T-OAR.py

### To Train:
* Train BFN
    * Prepare training data:

            Dataset name
            ├── train               # Training data
            │   ├── vis             # Visible images
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── ir              # Infrared images
            │   │   ├── ***.png
            │   │   └── ...
            ├── test                # Testing data
            │   ├── vis             # Visible images
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── ir              # Infrared images
            │   │   ├── ***.png
            │   │   └── ...
    * Run:

          python train_BFN.py

* Train T-OAR
    * Prepare training data:
      * Object Detection Dataset (M3FD_Detection):

            M3FD_Detection
            ├── ir               
            │   ├── train          
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── test           
            │   │   ├── ***.png
            │   │   └── ...
            ├── vi                
            │   ├── train              
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── test              
            │   │   ├── ***.png
            │   │   └── ...
            ├── labels                
            │   ├── train             
            │   │   ├── ***.txt
            │   │   └── ...
            │   ├── test              
            │   │   ├── ***.txt
            │   │   └── ...
      * Semantic Segmentation Dataset (FMB):
      
            FMB
            ├── train               
            │   ├── Infrared             
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── Visible              
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── Label              
            │   │   ├── ***.png
            │   │   └── ... 
            ├── test                
            │   ├── Infrared             
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── Visible              
            │   │   ├── ***.png
            │   │   └── ...
            │   ├── Label              
            │   │   ├── ***.png
            │   │   └── ...

      * Salient Object Detection Dataset (VT5000):

              VT5000
              ├── Train               
              │   ├── T_GRAY             
              │   │   ├── ***.png
              │   │   └── ...
              │   ├── RGB              
              │   │   ├── ***.png
              │   │   └── ...
              │   ├── GT              
              │   │   ├── ***.png
              │   │   └── ... 
              │   ├── Edge              
              │   │   ├── ***.png
              │   │   └── ...
              ├── Test                
              │   ├── T_GRAY             
              │   │   ├── ***.png
              │   │   └── ...
              │   ├── RGB              
              │   │   ├── ***.png
              │   │   └── ...
              │   ├── GT              
              │   │   ├── ***.png
              │   │   └── ... 
  * Run:

        torchrun --nproc_per_node 1 train_T-OAR.py

## Note
Due to the presence of numerous external links in the downstream task network code, it does not comply with the code release guidelines. Therefore, only the code for the method proposed in this paper has been made publicly available. The full code and pre-trained weights will be released after the paper is accepted.
