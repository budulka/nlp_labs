import requests
import bs4
import re 


def get_page(url: str) -> str:
    """
    Завантажує HTML-код сторінки за URL.
    
    Args:
        url (str): адреса сторінки
    
    Returns:
        str: HTML-код сторінки
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    return response.text


def parse(page: str):
    """
    Парсить HTML сторінки і знаходить елементи новин.
    
    Args:
        page (str): HTML-код сторінки
    
    Returns:
        list: список тегів <li> з класом 'l-entries__item'
    """
    soup = bs4.BeautifulSoup(page, 'lxml')
    quotes = soup.find_all('li', class_='l-entries__item')
    return quotes


def clean(row: str) -> str:
    """
    Очищає текст новини від службових слів, розривів рядків, ком та зайвих пробілів.
    
    Args:
        row (str): текст новини
    
    Returns:
        str: очищений текст
    """
    cleaned = row.replace("Дата публікації", "")
    cleaned = cleaned.replace("Категорія", "")
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    cleaned = cleaned.replace(',', '')
    return cleaned


def clean_content(quotes):
    """
    Очищає контент списку тегів новин, пропускає рекламу.
    
    Args:
        quotes (list): список тегів з HTML-контентом
    
    Returns:
        list: список очищених рядків новин
    """
    clean_quotes = []
    for q in quotes:
        c = clean(q.text)
        if c == "Реклама":
            continue
        clean_quotes.append(c)
    return clean_quotes


def save(rows, path: str):
    """
    Зберігає список рядків у файл.
    
    Args:
        rows (list): список рядків для збереження
        path (str): шлях до файлу
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for line in rows:
                f.write(line + '\n')
    except Exception as e:
        print("Could not save the results:", e)


def proccess_pages(pages: int, url: str) -> str:
    """
    Завантажує HTML код декількох сторінок і об'єднує їх в один рядок.
    
    Args:
        pages (int): кількість сторінок для завантаження
        url (str): базовий URL
    
    Returns:
        str: об'єднаний HTML код всіх сторінок
    """
    page = get_page(url)
    for i in range(2, pages+1):
        page += get_page(url + f"/page-{i}")
    return page


if __name__ == '__main__':
    url = 'https://tsn.ua/news'
    print('Обрано інформаційне джерело: ' + url)
    page = proccess_pages(10, url)
    quotes = parse(page)
    cleaned = clean_content(quotes)
    for q in cleaned:
        print(q)
    save(cleaned, 'result.txt')