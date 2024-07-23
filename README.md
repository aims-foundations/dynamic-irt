# LMS Scraping

## Installation

```
pip install -r requirements.txt
```

For `ChromeDriver` setup:
* From this [link](https://getwebdriver.com/chromedriver#stable), download the `ChromeDriver` version that adapts your machine's environment. Mine is Mac M2 (arm64). I choose the stable version to avoid cracking.
* Add `ChromeDriver` to PATH: place it in `/usr/local/bin` or any directory that is in your `PATH`. Then use the default code to access:

```
# chrome driver setup
chrome_options = Options()
chrome_options.add_argument('--headless')  # Runs Chrome in headless mode.
chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
driver = webdriver.Chrome(options=chrome_options)
```