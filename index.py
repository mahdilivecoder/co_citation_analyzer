import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import re
import sys
import os
import operator
from urllib.parse import unquote

# Persian/Arabic Text support for Matplotlib
import arabic_reshaper
from bidi.algorithm import get_display

# --- CONFIGURATION ---
CSV_FILENAME = 'input.csv'
REPORT_TXT = 'analysis_report.txt'
KEYWORDS_CSV = 'keywords_ranked.csv'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

STOP_WORDS = {
    'برای', 'توسط', 'باید', 'باشد', 'است', 'هستند', 'درباره', 'تماس', 'خانه', 
    'قیمت', 'تومان', 'ریال', 'مشاهده', 'ادامه', 'مطلب', 'در', 'به', 'از', 'که', 
    'می', 'این', 'را', 'با', 'های', 'آن', 'یک', 'شود', 'شده', 'کرد', 'نیز', 
    'بسیار', 'صورت', 'انجام', 'دارای', 'جهت', 'کاملا', 'مختلف', 'استفاده', 
    'بخش', 'مورد', 'طریق', 'همچنین', 'وجود', 'امکان', 'تمامی', 'کامل', 'ترین',
    'شامل', 'ارائه', 'محصول', 'خدمات', 'کلیک', 'کنید', 'بیشتر', 'نیست', 'ورق',
    'the', 'and', 'for', 'with'
}

def sanitize_url(url):
    url = url.strip()
    # Fix double slashes issue
    url = re.sub(r'(https?://)/+', r'\1', url)
    if not url.startswith('http'):
        url = 'https://' + url
    return unquote(url)

def fix_persian_text(text):
    """Reshapes text to render correctly in Matplotlib"""
    if not isinstance(text, str): return text
    # Check for Persian chars
    if re.search(r'[\u0600-\u06FF]', text):
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            return get_display(reshaped_text)
        except:
            return text
    return text

def get_user_input():
    print("==========================================")
    print("   SEO Strategy & Co-citation Tool        ")
    print("==========================================")
    print("Instructions: Enter URLs. Type 'end' to finish.\n")

    data_rows = []
    seen_urls = set()

    while True:
        raw_input = input(">> Website URL: ")
        if raw_input.lower().strip() in ['end', 'done', 'exit']: break
        
        url = sanitize_url(raw_input)
        if '.' not in url:
            print("Error: Invalid URL.")
            continue
            
        if url in seen_urls:
            print("Warning: URL already added.")
            continue

        print(f"   (Registered: {url})")
        keywords = input(">> Manual Keywords (Optional): ").strip()
        data_rows.append([url, keywords])
        seen_urls.add(url)
        print("--- Next ---\n")

    if not data_rows:
        sys.exit()

    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'keywords']) 
        writer.writerows(data_rows)
    print(f"\n[Success] Inputs saved to '{CSV_FILENAME}'.")

def get_auto_keywords(url, top_n=7):
    try:
        print(f" [...] Scanning: {url[:50]}...")
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
            
        text = soup.get_text()
        # Regex for words > 2 chars
        words = re.findall(r'[\u0600-\u06FF\u200ca-zA-Z]{2,}', text) 
        filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        return [word for word, count in Counter(filtered_words).most_common(top_n)]
    except Exception as e:
        print(f" [!] Error: {e}")
        return []

def get_smart_label(url):
    """
    Creates a label with:
    Line 1: Domain Name
    Line 2: Short Slug
    """
    # Remove protocol
    clean = url.replace('https://', '').replace('http://', '').replace('www.', '').rstrip('/')
    parts = clean.split('/')
    
    domain = parts[0]
    
    if len(parts) == 1:
        # Homepage
        return domain
    else:
        slug = parts[-1]
        if len(slug) > 15:
            slug = slug[:8] + "..." + slug[-5:]
        
        # Return formatted string with newline
        return f"{domain}\n({slug})"

def generate_analysis_report(G, url_list):
    print("\n--- Generating Implementation Plan... ---")
    
    all_nodes = list(G.nodes())
    site_nodes = [n for n in all_nodes if n in url_list]
    keyword_nodes = [n for n in all_nodes if n not in site_nodes]

    degrees = dict(G.degree(keyword_nodes))
    sorted_keywords = sorted(degrees.items(), key=operator.itemgetter(1), reverse=True)

    # Save CSV
    with open(KEYWORDS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['کلمه کلیدی', 'امتیاز', 'پیشنهاد'])
        for kw, score in sorted_keywords:
            sugg = "هدینگ اصلی (H1/H2)" if score > 2 else "متن پاراگراف"
            writer.writerow([kw, score, sugg])

    # Save Text Report
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("=== راهنمای عملی اجرای استراتژی (Action Plan) ===\n\n")
        
        f.write("1. کلمات کلیدی کانونی (Semantic Cores):\n")
        top_kws = [k[0] for k in sorted_keywords[:5]]
        f.write(f"   👉 {', '.join(top_kws)}\n\n")

        f.write("2. دستورالعمل پیاده‌سازی هم‌استنادی (Co-citation Guide):\n")
        f.write("   ✅ سناریوی لینک‌سازی:\n")
        f.write("   یک مقاله جدید بنویسید و در یک پاراگراف به منابع زیر لینک دهید:\n\n")
        
        f.write("   🔗 صفحات هدف برای لینک دادن:\n")
        for i, url in enumerate(site_nodes, 1):
            f.write(f"      {i}. {url}\n")
        
        f.write("\n   ✅ کلمات اتصال دهنده (Co-occurrence Context):\n")
        f.write("   در جملات اطراف لینک‌ها، از این جفت کلمات استفاده کنید:\n")
        
        edge_weights = nx.get_edge_attributes(G, 'weight')
        keyword_edges = {k: v for k, v in edge_weights.items() if k[0] in keyword_nodes and k[1] in keyword_nodes}
        sorted_edges = sorted(keyword_edges.items(), key=operator.itemgetter(1), reverse=True)
        
        for (w1, w2), w in sorted_edges[:5]:
            f.write(f"      - '{w1}' و '{w2}'\n")

    print(f" [File] Reports saved to '{REPORT_TXT}' and '{KEYWORDS_CSV}'.")

def analyze_data():
    try:
        df = pd.read_csv(CSV_FILENAME)
    except: return

    graph_data = {}
    print("--- Starting Analysis ---")
    
    url_list = [] 

    for index, row in df.iterrows():
        url = str(row['url']).strip()
        user_kw = str(row['keywords']).strip() if 'keywords' in df.columns else ""
        
        final_kws = []
        if user_kw and user_kw != 'nan' and user_kw != "":
            final_kws = [k.strip() for k in user_kw.split(',') if k.strip()]
        else:
            final_kws = get_auto_keywords(url)
        
        if final_kws:
            print(f" -> Found keywords for: {get_smart_label(url).replace(chr(10), ' ')}")
            graph_data[url] = final_kws
            url_list.append(url)

    G = nx.Graph()
    edge_weights = {}

    for url, kws in graph_data.items():
        for k in kws: G.add_edge(url, k, type='link')
        for pair in combinations(kws, 2):
            pair = tuple(sorted(pair))
            edge_weights[pair] = edge_weights.get(pair, 0) + 1
            
    for pair, w in edge_weights.items():
        G.add_edge(pair[0], pair[1], weight=w, type='co-occurrence')

    if len(G.nodes) > 0:
        generate_analysis_report(G, url_list)
        
        print("\n--- Plotting Graph... ---")
        plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(G, k=0.85, seed=42)
        
        site_nodes = [n for n in G.nodes if n in url_list]
        word_nodes = [n for n in G.nodes if n not in site_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=site_nodes, node_color='#ff9999', node_size=3000, label="Websites")
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='#99ff99', node_size=1500, label="Keywords")
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
        
        # --- LABEL GENERATION ---
        labels = {}
        for node in G.nodes():
            if node in site_nodes:
                # Use Smart Label (Domain + Slug)
                raw_label = get_smart_label(node)
                # Apply Persian Fix to the label (in case slug is Persian)
                labels[node] = fix_persian_text(raw_label)
            else:
                labels[node] = fix_persian_text(node)

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family='Tahoma')
        
        plt.title("SEO Co-citation & Keyword Network")
        plt.legend()
        plt.axis('off')
        plt.show()
    else:
        print("No data found.")

if __name__ == "__main__":
    get_user_input()
    analyze_data()