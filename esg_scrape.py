import requests
import json
from bs4 import BeautifulSoup
import re
import time

cookies = {
    '_gcl_au': '1.1.1341652015.1692613810',
    '_gid': 'GA1.2.1859801433.1692613813',
    '_gat': '1',
    'cookie_eu_consented': 'true',
    '__hstc': '95584451.ccf6f67c0d5cd4c18cce82888613e04b.1692613814305.1692613814305.1692613814305.1',
    'hubspotutk': 'ccf6f67c0d5cd4c18cce82888613e04b',
    '__hssrc': '1',
    '__hssc': '95584451.1.1692613814305',
    'AWSALB': 'Ox5ZEUqjngi+KbQHTLxbBOzdDEyI757lOx+S0C9+q3aZUx791y1u6zgiEss9ykkv26/TDHa+zeesu1CM201LEHDRgRSveiIe6R66CrCgjBHFq4st2BIJEtyyS8Ru',
    'AWSALBCORS': 'Ox5ZEUqjngi+KbQHTLxbBOzdDEyI757lOx+S0C9+q3aZUx791y1u6zgiEss9ykkv26/TDHa+zeesu1CM201LEHDRgRSveiIe6R66CrCgjBHFq4st2BIJEtyyS8Ru',
    '_csrhub-frontend_session': 'Z1NtaUVzMXNxZnEwaVZWYWZ6VHFaYjUrY2habWc2QzFoWk9YMnhwU0huZXZ6Z1JJM0RRRzJVU3U1dUd3cnFNenJoU2dCZDBrNWgwcWQ5RDVoZlI0S1NPYU84cm9jYUwxVElCcWV0Qm95bGphTEpqcW1zRGUxL1FoL3dzWUhZV3ZvWDUvc3NCbkhPUnlrUVlkcnM3aGcxZmh1TjdyeGNmOHY4YncrcnNoeXFDaGsrL1FDRzhmaGRzYUNTUC94eUIzLS1yZzE4eWFyOEZ0azh6d0ozWUlUV1hnPT0%3D--fbb68736a1d48c127b6da36da10e043454d27ecb',
    '_ga_GL1YLRCXF7': 'GS1.1.1692613810.1.1.1692613818.52.0.0',
    '_ga': 'GA1.2.1333055431.1692613810',
}

headers = {
    'Host': 'www.csrhub.com',
    'Sec-Ch-Ua-Mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.110 Safari/537.36',
    'Sec-Ch-Ua-Platform': '""',
    'Accept': '*/*',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://www.csrhub.com/search/country/India',
    # 'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.9',
    # Requests sorts cookies= alphabetically
    # 'Cookie': '_gcl_au=1.1.1341652015.1692613810; _gid=GA1.2.1859801433.1692613813; _gat=1; cookie_eu_consented=true; __hstc=95584451.ccf6f67c0d5cd4c18cce82888613e04b.1692613814305.1692613814305.1692613814305.1; hubspotutk=ccf6f67c0d5cd4c18cce82888613e04b; __hssrc=1; __hssc=95584451.1.1692613814305; AWSALB=Ox5ZEUqjngi+KbQHTLxbBOzdDEyI757lOx+S0C9+q3aZUx791y1u6zgiEss9ykkv26/TDHa+zeesu1CM201LEHDRgRSveiIe6R66CrCgjBHFq4st2BIJEtyyS8Ru; AWSALBCORS=Ox5ZEUqjngi+KbQHTLxbBOzdDEyI757lOx+S0C9+q3aZUx791y1u6zgiEss9ykkv26/TDHa+zeesu1CM201LEHDRgRSveiIe6R66CrCgjBHFq4st2BIJEtyyS8Ru; _csrhub-frontend_session=Z1NtaUVzMXNxZnEwaVZWYWZ6VHFaYjUrY2habWc2QzFoWk9YMnhwU0huZXZ6Z1JJM0RRRzJVU3U1dUd3cnFNenJoU2dCZDBrNWgwcWQ5RDVoZlI0S1NPYU84cm9jYUwxVElCcWV0Qm95bGphTEpqcW1zRGUxL1FoL3dzWUhZV3ZvWDUvc3NCbkhPUnlrUVlkcnM3aGcxZmh1TjdyeGNmOHY4YncrcnNoeXFDaGsrL1FDRzhmaGRzYUNTUC94eUIzLS1yZzE4eWFyOEZ0azh6d0ozWUlUV1hnPT0%3D--fbb68736a1d48c127b6da36da10e043454d27ecb; _ga_GL1YLRCXF7=GS1.1.1692613810.1.1.1692613818.52.0.0; _ga=GA1.2.1333055431.1692613810',
}

parsed = []

for i in range(58, 71):
    params = {
        'page': str(i),
    }

    response = requests.get('https://www.csrhub.com/api/v3/search:fully%20or%20partially%20rated%20and%20country%20is%20%22India%22/profile:467/public', params=params, cookies=cookies, headers=headers, verify=False)
    
    data = json.loads(response.text)
    print(data)
    for stock in data['data']:
        stock_name = stock['name']
        stock_name = stock_name.replace(' ', '-')
        stock_name = stock_name.replace('&', 'and')
        stock_name = stock_name.replace('.', '')
        stock_name = stock_name.replace("'", '')
        stock_name = stock_name.replace(",", '')
        stock_name = stock_name.replace("/", '')

        i=0
        while(i<1000):
            res = requests.get(f"https://www.csrhub.com/CSR_and_sustainability_information/{stock_name}")
            soup = BeautifulSoup(res.text, 'html.parser')

            parsed_data = {
                'name':stock_name,
                'esg': [],
                'esg_industry': []
            }

            s = soup.find('div', {'class': 'landing-gray-section landing-company-page-summary'})
            
            # Check if the div exists with class landing-offset-63
            if s:
                s = s.find('div', {'class': 'landing-offset-63'})
                if s:
                    break
            
            time.sleep(10)
            print('Retrying', stock_name)
            i+=1
        if i==1000:
            continue
        scripts = s.find_all('script')
        script_data = scripts[0].text.strip()
        m = re.search(r'small_chart_ratio_history_company = \[.*\];', script_data)
        if m:
            found = m.group(0)
            found = found.replace('small_chart_ratio_history_company = ', '')
            found = found.replace(';', '')
            found = json.loads(found)
            print(stock_name, found)
            for x in found:
                parsed_data['esg'].append(x)

        m = re.search(r'small_chart_ratio_history_industry = \[.*\];', script_data)
        if m:
            found = m.group(0)
            found = found.replace('small_chart_ratio_history_industry = ', '')
            found = found.replace(';', '')
            found = json.loads(found)
            print(stock_name, found)
            for x in found:
                parsed_data['esg_industry'].append(x)


        soup = soup.find('div', {'class': 'company-section_sheet'})
        # Get rows from the table inside div
        soup = soup.find('table').find_all('tr')

        # Iterate over rows
        for row in soup:
            # Get columns from the row
            cols = row.find_all('td')
            # Get the column containing the title
            title = cols[0].text.strip()
            # Get the column containing the value
            value = cols[1].text.strip()
            
            if title == 'Ticker:':
                parsed_data['ticker'] = value
            elif title == 'Industry:':
                parsed_data['industry'] = value
            
        parsed.append(parsed_data)

    with open('data.txt', 'w') as f:
        f.write(json.dumps(parsed, indent=4, sort_keys=True))
