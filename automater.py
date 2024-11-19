from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
import os


countries = ['México', 'Colombia', 'Costa Rica', 'Peru']
diminutive = {'México': 'MX', 'Colombia': 'CO', 'Costa Rica': 'CR', 'Peru': 'PE'
              }


def get_xlsx_files(folder_path):
    # List all files in the specified folder
    all_files = os.listdir(folder_path)

    # Filter and return only .xlsx files
    xlsx_files = [file for file in all_files if file.endswith('.xlsx')]

    return xlsx_files


def writeInBox(driver, input_xpath, textToInsert):
    while True:
        try:
            # Locate the input field using the provided XPath
            try:
                date_input = driver.find_element("xpath", input_xpath)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt caught! Exiting gracefully.")
                exit()
            except:
                date_input = driver.find_element(By.XPATH, input_xpath)

            # Clear the input field and enter the text
            date_input.clear()  # Clears any pre-existing value in the input
            date_input.send_keys(textToInsert)

            # Exit the loop if the input was successfully found and text inserted
            break

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught! Exiting gracefully.")
            exit()
        except NoSuchElementException:
            print("Element not found, waiting for 5 seconds and retrying...")
            sleep(5)
        except StaleElementReferenceException:
            print("Stale element reference, waiting for 5 seconds and retrying...")
            sleep(5)


def read_first_two_lines(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()  # Read first line
            second_line = file.readline().strip()  # Read second line
            return first_line, second_line
    except FileNotFoundError:
        return None
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Exiting gracefully.")
        return None


def click_button(driver, button_xpath):
    try:
        button = driver.find_element(By.XPATH, button_xpath)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Exiting gracefully.")
        exit()
    except:
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, button_xpath)))

    driver.execute_script(
        "arguments[0].scrollIntoView(); arguments[0].click();", button)

    return True


def checkCredentials(driver, file_path='credentials.txt'):
    result = read_first_two_lines(file_path)

    if result is None:
        print("File not found.")
    else:
        while True:
            try:
                first_line, second_line = result
                writeInBox(driver, '//*[@id="username"]', first_line)
                writeInBox(driver, '//*[@id="password"]', second_line)
                while True:
                    try:
                        click_button(driver, '//*[@id="submit"]')
                        break
                    except KeyboardInterrupt:
                        print("\nKeyboardInterrupt caught! Exiting gracefully.")
                        exit()
                    except:
                        print('Submit button not found')
                break
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt caught! Exiting gracefully.")
                exit()
            except:
                print('Credentials found, trying again the login')
                sleep(5)
    return


def check_element(driver, xpath):
    try:
        driver.find_element("xpath", xpath)
        return True
    except NoSuchElementException:
        try:
            driver.find_element(By.XPATH, xpath)
            return True
        except:
            pass
        return False
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Exiting gracefully.")
        exit()


def is_page_loading(driver):
    return driver.execute_script("return document.readyState") != "complete"


def enterDiDiDashboard(driver):
    # XPath for the submit button
    checkCredentials(driver)
    xpath = '//*[@id="submit"]'
    while check_element(driver, xpath) and 'https://me.didiglobal.com/project/stargate-auth' in driver.current_url:
        print("Logging in...")
        sleep(5)
    sleep(5)


def start():
    # Setup ChromeOptions to suppress logs
    chrome_options = Options()
    # Suppress logs (INFO, WARNING, ERROR, FATAL)
    chrome_options.add_argument("--log-level=3")

    # Additional option to suppress DevTools and SSL errors
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--ignore-certificate-errors")

    driver = webdriver.Chrome(options=chrome_options)
    return driver


def doAllProcess(driver, country, filename):
    print(f'Try to upload file {filename} for {country}')

    section = driver.find_element(
        By.CLASS_NAME, "gtr-header-tools-country-time")
    driver.execute_script("arguments[0].click();", section)
    sleep(5)
    country_span = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, f"//span[text()='{country}']"))
    )

    # Scroll into view and click using JavaScript
    driver.execute_script(
        "arguments[0].scrollIntoView(); arguments[0].click();", country_span)

    click_button(
        driver, '//*[@id="gtr-layout-nav_function"]/div[1]/div/div[3]/span/button[2]')

    sleep(5)
    upload_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')

    upload_input.send_keys(
        f'{os.path.dirname(os.path.abspath(__file__))}/files_to_upload/{filename}')

    sleep(5)

    click_button(
        driver, '//*[@id="pane-batch-operation"]/div/div[3]/button[1]')
    sleep(5)
    driver.get(
        'https://gattaran.didi-food.com/v2/gtr_crm/bizopp/updateSignStores/batch-operation')
    sleep(5)
    print('In theory, file uploaded')


def doSelenium():
    driver = start()
    # Open the target webpage
    driver.get(
        'https://gattaran.didi-food.com/v2/gtr_crm/bizopp/updateSignStores/batch-operation')
    sleep(5)
    enterDiDiDashboard(driver)
    print('acabé big loggin function')
    i = 0
    while i < 5:
        try:
            click_button(
                driver, '//*[@id="driver-popover-item"]/div[4]/span[2]/button[2]')
            i += 1
            sleep(3)
        except:
            print('Fallé D:')
            raise ValueError
    print('Ya acabé el tutorial')
    sleep(3)
    files = get_xlsx_files('files_to_upload')
    for country in countries:
        importantFiles = [x for x in files if diminutive[country] in x]
        for file in importantFiles:
            doAllProcess(driver, country, file)


if __name__ == '__main__':
    doSelenium()
