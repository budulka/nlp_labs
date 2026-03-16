import requests
import bs4
import re 

def get_page(url : str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    return response.text

def parse(page):
    soup = bs4.BeautifulSoup(page, 'lxml')
    quotes = soup.find_all('li', class_='l-entries__item')
    return quotes

def clean(row : str):
    cleaned = row.replace("Дата публікації", "")
    cleaned = cleaned.replace("Категорія", "")
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    cleaned = cleaned.replace(',', '')
    
    return cleaned

def clean_content(quotes):
    clean_quotes = []
    for q in quotes:
        c = clean(q.text)
        if c == "Реклама":
            continue
        clean_quotes.append(c)
    return clean_quotes

def save(rows, path):
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
       


    