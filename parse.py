import requests
import bs4
import re 



def get_page(url : str):
    """Отримання HTML сторінки
        url: str - посилання на сторінку, яку треба отримати
        Повертає код сторінки як строку
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    return response.text

def parse(page):
    """Парсинг сторінки
        page - Код сторінки
        Повертає список елементів класу l-entries__item
    """
    soup = bs4.BeautifulSoup(page, 'lxml')
    quotes = soup.find_all('li', class_='l-entries__item')
    return quotes

def clean(row : str):
    """Очищення елементів
        row : str - текст елементу
        Повертає список елементів класу l-entries__item
    """
    cleaned = row.replace("Дата публікації", "")
    cleaned = cleaned.replace("Категорія", "")
    cleaned = cleaned.replace('\n', ' ')В
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    cleaned = cleaned.replace(',', '')
    
    return cleaned

def clean_content(quotes):
    """ Очищення списку елементів 
        для кожного елементу викликаємо clean
        quotes - список елементів
        Повертає список очищених елементів

    """
    clean_quotes = []
    for q in quotes:
        c = clean(q.text)
        if c == "Реклама":
            continue
        clean_quotes.append(c)
    return clean_quotes

def save(rows, path):
    """"Збереження у файл"""
    try:
        with open(path, 'w') as f:
            for line in rows:
                f.write(line + '\n')
    except Exception as e:
        print("Could not save the results")
    

if __name__ == '__main__':
    url = 'https://tsn.ua/news'
    print('Обрано інформаційне джерело: ' + url)
    page = get_page(url)
    quotes = parse(page)
    cleaned = clean_content(quotes)
    for q in cleaned:
        print(q)
    save(cleaned, 'result.txt')
       


    