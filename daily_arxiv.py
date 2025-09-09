import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
import urllib3
from typing import List, Dict, Tuple, Optional, Pattern

# Disable SSL warnings when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"

# ----------------------------
# Keyword-based relevance utils
# ----------------------------

def _normalize_kw(kw: str) -> str:
    return kw.strip().lower()

def _word_regex(kw: str) -> Pattern:
    """Build a regex for whole-word matching when keyword is a single token.
    Fallback to simple escaped substring for phrases.
    """
    kw = kw.strip()
    # If multi-word or contains non-word chars, just escape and search as substring (case-insensitive)
    if len(kw.split()) > 1 or re.search(r"[^\w]", kw):
        return re.compile(re.escape(kw), re.IGNORECASE)
    # Single token -> whole word
    return re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)

def _count_hits(text: str, terms: List[str]) -> int:
    if not text or not terms:
        return 0
    count = 0
    for t in terms:
        if not t:
            continue
        if _word_regex(t).search(text) is not None:
            count += 1
    return count

def _is_relevant_for_topic(title: str,
                           abstract: str,
                           topic_rules: Optional[Dict]) -> Tuple[bool, int, Dict]:
    """Decide if a paper is relevant to a topic using title/abstract keyword rules.

    Backward compatible with existing config schema:
      - If topic_rules has 'filters': use as include-any terms
      - Optional advanced keys:
          include: { any: [...], all: [...] }
          exclude: [...]
          min_score: int (default computed)
          title_weight: int (default 2)

    Returns (is_relevant, score, details)
    """
    if topic_rules is None:
        return True, 0, {"reason": "no-rules"}

    text_title = title or ""
    text_abs = abstract or ""

    include_any = topic_rules.get("include", {}).get("any") if isinstance(topic_rules.get("include"), dict) else None
    include_all = topic_rules.get("include", {}).get("all") if isinstance(topic_rules.get("include"), dict) else None
    # Backward-compat: treat 'filters' as include-any
    if not include_any:
        include_any = topic_rules.get("filters", []) or []
    exclude_terms = topic_rules.get("exclude", []) or []

    # Normalize lists
    include_any = [_normalize_kw(k) for k in include_any]
    include_all = [_normalize_kw(k) for k in (include_all or [])]
    exclude_terms = [_normalize_kw(k) for k in exclude_terms]

    title_weight = int(topic_rules.get("title_weight", 2))
    min_score = topic_rules.get("min_score")

    # Heuristic default threshold: if only a few specific filters, 1; if many generic filters, 2
    if min_score is None:
        min_score = 1 if len(include_any) <= 4 else 2

    # Exclusion check first
    excluded = _count_hits(text_title, exclude_terms) + _count_hits(text_abs, exclude_terms) > 0
    if excluded:
        return False, 0, {"reason": "exclude-hit"}

    # Compute score
    title_hits_any = _count_hits(text_title, include_any)
    abs_hits_any = _count_hits(text_abs, include_any)
    score_any = title_weight * title_hits_any + abs_hits_any

    # All-of check (must match all terms somewhere in title+abstract)
    all_ok = True
    if include_all:
        for t in include_all:
            if _word_regex(t).search(text_title) is None and _word_regex(t).search(text_abs) is None:
                all_ok = False
                break

    is_relevant = (score_any >= min_score) and all_ok
    details = {
        "title_hits": title_hits_any,
        "abs_hits": abs_hits_any,
        "score": score_any,
        "min_score": min_score,
        "all_ok": all_ok,
    }
    return is_relevant, score_any, details

def _safe_json_loads(content: str) -> dict:
    """
    Safely load JSON content.
    - If empty -> return {}
    - If valid JSON -> return parsed object
    - If concatenated JSON objects (Extra data) -> parse the first object and ignore the rest
    - On any error -> log and return {}
    """
    if not content or content.strip() == "":
        return {}
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
        # If it's not a dict, return empty to keep expected shape
        logging.warning("JSON root is not an object; resetting to empty dict.")
        return {}
    except json.JSONDecodeError as e:
        # Try to decode the first JSON object if file contains concatenated JSON
        try:
            decoder = json.JSONDecoder()
            obj, end = decoder.raw_decode(content)
            if isinstance(obj, dict):
                logging.warning(f"JSON contained extra data after first object at pos {end}; ignoring trailing content.")
                return obj
            logging.warning("First decoded JSON value is not an object; resetting to empty dict.")
            return {}
        except Exception:
            logging.error(f"Failed to parse JSON file: {e}. Resetting to empty dict.")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON content: {e}. Resetting to empty dict.")
        return {}

def _read_json_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _safe_json_loads(f.read())
    except FileNotFoundError:
        return {}
    except Exception as e:
        logging.error(f"Error reading JSON file '{path}': {e}")
        return {}

def load_config(config_file:str) -> dict:
    '''
    config_file: input config file path
    return: a dict of configuration
    '''
    # make filters pretty
    def pretty_filters(**config) -> dict:
        keywords = dict()
        EXCAPE = '\"'
        QUOTA = '' # NO-USE
        OR = ' OR ' # Fixed: added spaces around OR
        def parse_filters(filters:list):
            ret = ''
            for idx in range(0,len(filters)):
                filter = filters[idx]
                if len(filter.split()) > 1:
                    ret += (EXCAPE + filter + EXCAPE)  
                else:
                    ret += (QUOTA + filter + QUOTA)   
                if idx != len(filters) - 1:
                    ret += OR
            return ret
        for k,v in config['keywords'].items():
            keywords[k] = parse_filters(v['filters'])
        return keywords
    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader) 
        config['kv'] = pretty_filters(**config)
        logging.info(f'config = {config}')
    return config 

def get_authors(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output
def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output    
import requests

def get_code_link(qword:str) -> Optional[str]:
    """
    This short function was auto-generated by ChatGPT. 
    I only renamed some params and added some comments.
    @param qword: query string, eg. arxiv ids and paper titles
    @return paper_code in github: string, if not found, return None
    """
    try:
        # query = f"arxiv:{arxiv_id}"
        query = f"{qword}"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc"
        }
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "asr-tts-paper-daily-bot"
        }
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        r = requests.get(github_url, params=params, headers=headers, timeout=10)
        r.raise_for_status()  # Raise an exception for bad status codes
        results = r.json()
        code_link = None
        if "total_count" in results and results["total_count"] > 0:
            code_link = results["items"][0]["html_url"]
        return code_link
    except Exception as e:
        logging.warning(f"GitHub search failed for query '{qword}': {e}")
        return None
  
def get_daily_papers(topic,query="slam", max_results=2, topic_rules: Optional[Dict] = None):
    """
    @param topic: str
    @param query: str
    @return paper_with_code: dict
    """
    # output 
    content = dict() 
    content_to_web = dict()
    
    try:
        # Use the new Client API instead of deprecated Search
        client = arxiv.Client()
        search_engine = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )

        paper_count = 0
        for result in client.results(search_engine):
            if paper_count >= max_results:
                break
            
            paper_id            = result.get_short_id()
            paper_title         = result.title
            paper_url           = result.entry_id
            code_url            = base_url + paper_id #TODO
            paper_abstract      = result.summary.replace("\n"," ")
            paper_authors       = get_authors(result.authors)
            paper_first_author  = get_authors(result.authors,first_author = True)
            primary_category    = result.primary_category
            publish_time        = result.published.date()
            update_time         = result.updated.date()
            comments            = result.comment

            # Relevance filtering using title+abstract
            is_rel, rel_score, rel_details = _is_relevant_for_topic(
                paper_title, paper_abstract, topic_rules
            )
            if not is_rel:
                logging.info(f"Skip (irrelevant) [{topic}] score={rel_details.get('score')}, title='{paper_title[:80]}'")
                continue
            paper_count += 1

            logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

            # eg: 2108.09112v1 -> 2108.09112
            ver_pos = paper_id.find('v')
            if ver_pos == -1:
                paper_key = paper_id
            else:
                paper_key = paper_id[0:ver_pos]    
            paper_url = arxiv_url + 'abs/' + paper_key
            
            try:
                # source code link with SSL verification and timeout
                repo_url = None
                try:
                    # First try with normal SSL verification
                    r = requests.get(code_url, timeout=10).json()
                    if "official" in r and r["official"]:
                        repo_url = r["official"]["url"]
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                    # Fallback: try without SSL verification
                    try:
                        r = requests.get(code_url, timeout=10, verify=False).json()
                        if "official" in r and r["official"]:
                            repo_url = r["official"]["url"]
                    except:
                        # If paperswithcode fails, try GitHub search as fallback
                        repo_url = get_code_link(paper_title)
                        if repo_url is None:
                            repo_url = get_code_link(paper_key)
                except:
                    # If paperswithcode fails, try GitHub search as fallback
                    repo_url = get_code_link(paper_title)
                    if repo_url is None:
                        repo_url = get_code_link(paper_key)
                        
                if repo_url is not None:
                    content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|**[link]({})**|\n".format(
                        update_time, paper_title, paper_first_author, paper_key, paper_url, repo_url)
                    content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({}), Code: **[{}]({})**".format(
                        update_time, paper_title, paper_first_author, paper_url, paper_url, repo_url, repo_url)

                else:
                    content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|null|\n".format(
                        update_time, paper_title, paper_first_author, paper_key, paper_url)
                    content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({})".format(
                        update_time, paper_title, paper_first_author, paper_url, paper_url)

                # TODO: select useful comments
                comments = None
                if comments != None:
                    content_to_web[paper_key] += f", {comments}\n"
                else:
                    content_to_web[paper_key] += f"\n"

            except Exception as e:
                logging.error(f"exception: {e} with id: {paper_key}")
                
    except Exception as e:
        logging.error(f"Error fetching papers for topic '{topic}': {e}")
        # Return empty data if there's a complete failure
        pass

    data = {topic:content}
    data_web = {topic:content_to_web}
    return data,data_web 

def update_paper_links(filename):
    '''
    weekly update paper links in json file 
    '''
    def parse_arxiv_string(s):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
        arxiv_id = re.sub(r'v\d+', '', arxiv_id)
        return date,title,authors,arxiv_id,code

    m = _read_json_file(filename)
    json_data = m.copy()

    for keywords, v in json_data.items():
        logging.info(f'keywords = {keywords}')
        for paper_id, contents in v.items():
            contents = str(contents)

            update_time, paper_title, paper_first_author, paper_url, code_url = parse_arxiv_string(contents)

            contents = "|{}|{}|{}|{}|{}|\n".format(update_time, paper_title, paper_first_author, paper_url, code_url)
            json_data[keywords][paper_id] = str(contents)
            logging.info(f'paper_id = {paper_id}, contents = {contents}')

            valid_link = False if '|null|' in contents else True
            if valid_link:
                continue
            try:
                code_url = base_url + paper_id  # TODO
                repo_url = None
                try:
                    # First try with normal SSL verification
                    r = requests.get(code_url, timeout=10).json()
                    if "official" in r and r["official"]:
                        repo_url = r["official"]["url"]
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                    # Fallback: try without SSL verification
                    try:
                        r = requests.get(code_url, timeout=10, verify=False).json()
                        if "official" in r and r["official"]:
                            repo_url = r["official"]["url"]
                    except:
                        pass  # Skip if both methods fail

                if repo_url is not None:
                    new_cont = contents.replace('|null|', f'|**[link]({repo_url})**|')
                    logging.info(f'ID = {paper_id}, contents = {new_cont}')
                    json_data[keywords][paper_id] = str(new_cont)

            except Exception as e:
                logging.error(f"exception: {e} with id: {paper_id}")
    # dump to json file
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

def update_json_file(filename,data_dict):
    '''
    daily update json file using data_dict
    '''
    m = _read_json_file(filename)
    json_data = m.copy() 
    
    # update papers in each keywords         
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename,"w", encoding='utf-8') as f:
        json.dump(json_data,f, ensure_ascii=False, indent=2)
    
def json_to_md(filename,md_filename,
               task = '',
               to_web = False, 
               use_title = True, 
               use_tc = True,
               show_badge = True,
               use_b2t = True):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """
    def pretty_math(s:str) -> str:
        ret = ''
        match = re.search(r"\$.*\$", s)
        if match == None:
            return s
        math_start,math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]: space_trail = ' ' 
        if s[math_end:][0] != ' ' and '*' != s[math_end:][0]: space_leading = ' ' 
        ret += s[:math_start] 
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}' 
        ret += s[math_end:]
        return ret
    
    def make_anchor(text: str) -> str:
        """Convert text to GitHub-style anchor link"""
        # Remove emojis and keep only the text part
        # GitHub ignores emojis when creating anchors
        text_only = re.sub(r'[^\w\s]', '', text).strip()
        # Convert to lowercase and replace spaces with hyphens
        anchor = text_only.lower().replace(' ', '-')
        return anchor
  
    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-','.')
    
    data = _read_json_file(filename)

    # clean README.md if daily already exist else create it
    with open(md_filename,"w+", encoding='utf-8') as f:
        pass

    # write data into README.md
    with open(md_filename,"a+", encoding='utf-8') as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")
        
        if show_badge == True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n")
            f.write(f"[![GitHub Pages][pages-shield]][pages-url]\n\n")    
                
        if use_title == True:
            if to_web == False:
                # Enhanced README header
                f.write("<div align=\"center\">\n\n")
                f.write("# 🎯 ASR-TTS Paper Daily\n\n")
                f.write("*Automatically curated collection of the latest research papers in Speech & Language Technology*\n\n")
                f.write("📅 **Updated on " + DateNow + "**\n\n")
                f.write("---\n\n")
                f.write("</div>\n\n")
            else:
                f.write("# 🎯 ASR-TTS Paper Daily\n\n")
                f.write("*Automatically curated collection of the latest research papers in Speech & Language Technology*\n\n")
                f.write("📅 **Updated on " + DateNow + "**\n\n")
        else:
            f.write("> 📅 Updated on " + DateNow + "\n")

        # Enhanced description
        f.write("## 🌟 About This Repository\n\n")
        f.write("This repository provides a **daily-updated collection** of the latest research papers from arXiv in the following domains:\n\n")
        f.write("- 🎤 **Automatic Speech Recognition (ASR)**\n")
        f.write("- 🗣️ **Text-to-Speech (TTS)**\n") 
        f.write("- 🌐 **Machine Translation**\n")
        f.write("- ⚡ **Small Language Models**\n")
        f.write("- 🔄 **Data Augmentation**\n")
        f.write("- 🎨 **Synthetic Generation**\n\n")
        f.write("> 📖 Usage instructions: [here](./docs/README.md#usage) | 🌐 Web version: [GitHub Pages](https://nickdee96.github.io/ASR-TTS-paper-daily/)\n\n")
        f.write("> 💡 This page is inspired by [cv-arxiv-daily](https://github.com/Vincentqyw/cv-arxiv-daily)\n\n")

        #Add: table of contents
        if use_tc == True:
            # Heading id only for GitHub Pages (kramdown). GitHub README ignores {#id}.
            if to_web:
                f.write("## 📚 Table of Contents {#table-of-contents}\n\n")
            else:
                # Use emoji-free heading in README to produce predictable anchors
                f.write("## Table of Contents\n\n")
            
            # Add category emojis
            category_emojis = {
                "ASR": "🎤",
                "TTS": "🗣️", 
                "Machine Translation": "🌐",
                "Small Language Models": "⚡",
                "Data Augmentation": "🔄",
                "Synthetic Generation": "🎨"
            }
            
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                emoji = category_emojis.get(keyword, "📄")
                # Create consistent anchor link
                # For README (non-web), use emoji-free slug to satisfy linters
                if to_web:
                    anchor = make_anchor(f"{emoji} {keyword}")
                else:
                    anchor = make_anchor(f"{keyword}")
                f.write(f"- {emoji} **[{keyword}](#{anchor})**\n")
            f.write("\n---\n\n")
        
        # Category emojis for section headers
        category_emojis = {
            "ASR": "🎤",
            "TTS": "🗣️", 
            "Machine Translation": "🌐",
            "Small Language Models": "⚡",
            "Data Augmentation": "🔄",
            "Synthetic Generation": "🎨"
        }
        
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            
            # Enhanced section header with emoji and styling
            emoji = category_emojis.get(keyword, "📄")
            if to_web:
                # Explicit id for GitHub Pages (can include emoji in text)
                anchor = make_anchor(f"{emoji} {keyword}")
                f.write(f"## {emoji} {keyword} {{#{anchor}}}\n\n")
            else:
                # Emoji-free heading for GitHub README to ensure stable anchors and pass linters
                f.write(f"## {keyword}\n\n")
            
            # Add paper count
            paper_count = len(day_content)
            f.write(f"*📊 {paper_count} papers*\n\n")

            if use_title == True :
                if to_web == False:
                    # Enhanced table with better styling
                    f.write("<div align=\"center\">\n\n")
                    f.write("| 📅 **Publish Date** | 📝 **Title** | 👥 **Authors** | 📄 **PDF** | 💻 **Code** |\n")
                    f.write("|:---:|:---|:---:|:---:|:---:|\n")
                else:
                    f.write("| 📅 **Publish Date** | 📝 **Title** | 👥 **Authors** | 📄 **PDF** | 💻 **Code** |\n")
                    f.write("|:---------|:-----------------------|:---------|:------|:------|\n")

            # sort papers by date
            day_content = sort_papers(day_content)
        
            for _,v in day_content.items():
                if v is not None:
                    f.write(pretty_math(v)) # make latex pretty

            # Close table div for non-web version
            if use_title == True and to_web == False:
                f.write("\n</div>\n\n")
            else:
                f.write(f"\n")
            
            #Add: back to top with styling
            if use_b2t:
                f.write("<div align=\"right\">\n\n")
                if to_web:
                    toc_anchor = make_anchor("📚 Table of Contents")
                else:
                    toc_anchor = make_anchor("Table of Contents")
                f.write(f"*[⬆️ Back to Top](#{toc_anchor})*\n\n")
                f.write("</div>\n\n")
                f.write("---\n\n")
            
        if show_badge == True:
            # Add footer section
            f.write("---\n\n")
            f.write("## 🤝 Contributing\n\n")
            f.write("Contributions are welcome! Please feel free to submit issues or pull requests.\n\n")
            f.write("## ⭐ Star History\n\n")
            f.write("If you find this repository useful, please consider giving it a star!\n\n")
            
            # Updated badge links for your repository
            f.write((f"[contributors-shield]: https://img.shields.io/github/"
                     f"contributors/nickdee96/ASR-TTS-paper-daily.svg?style=for-the-badge\n"))
            f.write((f"[contributors-url]: https://github.com/nickdee96/"
                     f"ASR-TTS-paper-daily/graphs/contributors\n"))
            f.write((f"[forks-shield]: https://img.shields.io/github/forks/nickdee96/"
                     f"ASR-TTS-paper-daily.svg?style=for-the-badge\n"))
            f.write((f"[forks-url]: https://github.com/nickdee96/"
                     f"ASR-TTS-paper-daily/network/members\n"))
            f.write((f"[stars-shield]: https://img.shields.io/github/stars/nickdee96/"
                     f"ASR-TTS-paper-daily.svg?style=for-the-badge\n"))
            f.write((f"[stars-url]: https://github.com/nickdee96/"
                     f"ASR-TTS-paper-daily/stargazers\n"))
            f.write((f"[issues-shield]: https://img.shields.io/github/issues/nickdee96/"
                     f"ASR-TTS-paper-daily.svg?style=for-the-badge\n"))
            f.write((f"[issues-url]: https://github.com/nickdee96/"
                     f"ASR-TTS-paper-daily/issues\n"))
            f.write((f"[pages-shield]: https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?style=for-the-badge&logo=github\n"))
            f.write((f"[pages-url]: https://nickdee96.github.io/ASR-TTS-paper-daily/\n\n"))
                
    logging.info(f"{task} finished")        

def demo(**config):
    # TODO: use config
    data_collector = []
    data_collector_web= []
    
    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']

    b_update = config['update_paper_links']
    logging.info(f'Update Paper Link = {b_update}')
    if config['update_paper_links'] == False:
        logging.info(f"GET daily papers begin")
        for topic, keyword in keywords.items():
            logging.info(f"Keyword: {topic}")
            # Pull raw rules from config['keywords'][topic] for relevance filtering
            topic_rules = None
            try:
                topic_rules = config.get('keywords', {}).get(topic, None)
            except Exception:
                topic_rules = None
            data, data_web = get_daily_papers(topic, query = keyword,
                                            max_results = max_results,
                                            topic_rules = topic_rules)
            data_collector.append(data)
            data_collector_web.append(data_web)
            print("\n")
        logging.info(f"GET daily papers end")

    # 1. update README.md file
    if publish_readme:
        json_file = config['json_readme_path']
        md_file   = config['md_readme_path']
        # update paper links
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:    
            # update json data
            update_json_file(json_file,data_collector)
        # json data to markdown
        json_to_md(json_file,md_file, task ='Update Readme', \
            show_badge = show_badge)

    # 2. update docs/index.md file (to gitpage)
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file   = config['md_gitpage_path']
        # TODO: duplicated update paper links!!!
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:    
            update_json_file(json_file,data_collector)
        json_to_md(json_file, md_file, task ='Update GitPage', \
            to_web = True, show_badge = show_badge, \
            use_tc=False, use_b2t=False)

    # 3. Update docs/wechat.md file
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file   = config['md_wechat_path']
        # TODO: duplicated update paper links!!!
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:    
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task ='Update Wechat', \
            to_web=False, use_title= False, show_badge = show_badge)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    parser.add_argument('--update_paper_links', default=False,
                        action="store_true",help='whether to update paper links etc.')                        
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links':args.update_paper_links}

    demo(**config)
