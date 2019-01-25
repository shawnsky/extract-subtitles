# Subtitles Extraction
This project is used to extract subtitles from the video. First, the key frames is extracted from the video, and then the subtitle area of the frame picture is cropped, and the text is recognized by the OCR. Extract key frames from [Amanpreet Walia](https://github.com/amanwalia92).

## Getting Started
### Install following dependences

- numpy (find here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
- matplotlib `pip install Matplotlib`
- scipy
- opencv-python
- pytesseract `pip install pytesseract`

### Install Tesseract Ocr

[Download](https://github.com/UB-Mannheim/tesseract/wiki) and run it, select language support you want. Then modify the `../site-packages/pytesseract/pytesseract.py` :

`tesseract_cmd = 'X:\\Tesseract-OCR\\tesseract.exe'` (your install path)

### Run

```
λ python extract_subtitles.py <videopath> <Paremeter to select how many frames you want>
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

