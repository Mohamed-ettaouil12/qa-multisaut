# Exemples SFT - J8 (10 premiers)

## Exemple 1 (qid: 5ae381695542992f92d822f1, aléatoire)

**Question:** Carl Bialik is an American journalist, who on February 6, 2017, was named Data Science Editor of Yelp, working on Yelpblog, in late 2013, Bialik was hired by which American statistician and writer who analyzes baseball (see sabermetrics) and elections,at FiveThirtyEight.com?

**Réponse gold:** Nate Silver

**Triplets dans contexte (9):**
  [1] (maggie koerth-baker, 1981, date of birth)
  [2] (dean oliver, february 6, 1969, date of birth)
  [3] (dean oliver, february 6, 1969, date of birth)
  [4] (cook's distance, r. dennis cook, named after)
  [5] (sabermetrics, baseball, sport) ✓
  [6] (pete palmer, january 30, 1938, date of birth)
  [7] (denver nuggets, nba, league)
  [8] (nate silver, has_info, sport) ✓
  [9] (dean oliver (statistician), has_info, date of birth)

**Prompt:**
```
Question: Carl Bialik is an American journalist, who on February 6, 2017, was named Data Science Editor of Yelp, working on Yelpblog, in late 2013, Bialik was hired by which American statistician and writer who analyzes baseball (see sabermetrics) and elections,at FiveThirtyEight.com?

Contexte (triplets de connaissance):
[1] (sabermetrics, baseball, sport) ✓
[2] (nate silver, has_info, sport) ✓
[3] (maggie koerth-baker, 1981, date of birth)
[4] (dean oliver, february 6, 1969, date of birth)
[5] (dean oliver, february 6, 1969, date of birth)
[6] (cook's distance, r. dennis cook, named after) — Source: " It is named after the American statistician R. Dennis Cook, who introduced the concept in 1977."
[7] (pete palmer, january 30, 1938, date of birth)
[8] (denver nuggets, nba, league) — Source: " TruMedia Networks is an engineering firm specializing in sports analytics solutions for leagues, franchises and media partners, and says of Oliver in a press release announcing Dean Oliver's hiring, "
[9] (dean oliver (statistician), has_info, date of birth)

Réponse:
```

**Cible (completion):**
```
Nate Silver Sources: [5][8]
```

---

## Exemple 2 (qid: 5a7b46de55429927d897bf6d, cosinus)

**Question:** Boston and Skegness is represented in the House of Commons by which politician and former journalist?

**Réponse gold:** Matt Warman

**Triplets dans contexte (5):**
  [1] (boston and skegness (uk parliament constituency), has_info, part of) ✓
  [2] (sir henry cavendish, liberal, member of political party)
  [3] (uk parliament, house of commons, has part) ✓
  [4] (house of commons, uk parliament, part of) ✓
  [5] (baron waterpark, has_info, member of political party)

**Prompt:**
```
Question: Boston and Skegness is represented in the House of Commons by which politician and former journalist?

Contexte (triplets de connaissance):
[1] (boston and skegness (uk parliament constituency), has_info, part of) ✓
[2] (uk parliament, house of commons, has part) ✓
[3] (house of commons, uk parliament, part of) ✓
[4] (sir henry cavendish, liberal, member of political party)
[5] (baron waterpark, has_info, member of political party)

Réponse:
```

**Cible (completion):**
```
Matt Warman Sources: [1][3][4]
```

---

## Exemple 3 (qid: 5abee7c75542993fe9a41d91, aléatoire)

**Question:** Lorenzo Doss was on the championship team that played in what 2015 game?

**Réponse gold:** Super Bowl 50

**Triplets dans contexte (14):**
  [1] (2003 nfl draft, national football league, organizer)
  [2] (joey haywood, basketball, sport)
  [3] (national basketball association, basketball, sport)
  [4] (mike doss, has_info, league)
  [5] (cameron dollar, university of california, los angeles, educated at)
  [6] (cornerback, american football, sport) ✓
  [7] (list of nbl canada all-stars, has_info, sport)
  [8] (tom thacker (basketball), has_info, sport)
  [9] (national football league, american football, sport) ✓
  [10] (cincinnati royals, basketball, sport)
  [11] (lorenzo doss, has_info, sport) ✓
  [12] (tulane, american football, sport) ✓
  [13] (american basketball association, basketball, sport)
  [14] (indianapolis colts, national football league, league)

**Prompt:**
```
Question: Lorenzo Doss was on the championship team that played in what 2015 game?

Contexte (triplets de connaissance):
[1] (cornerback, american football, sport) ✓
[2] (national football league, american football, sport) ✓
[3] (lorenzo doss, has_info, sport) ✓
[4] (tulane, american football, sport) ✓
[5] (2003 nfl draft, national football league, organizer)
[6] (joey haywood, basketball, sport)
[7] (national basketball association, basketball, sport)
[8] (mike doss, has_info, league) — Source: "Michael Allen Doss (born March 24, 1981) is an American former college and professional football player who was a safety in the National Football League (NFL) for six seasons during the early 2000s."
[9] (cameron dollar, university of california, los angeles, educated at)
[10] (list of nbl canada all-stars, has_info, sport)
[11] (tom thacker (basketball), has_info, sport)
[12] (cincinnati royals, basketball, sport)
[13] (american basketball association, basketball, sport)
[14] (indianapolis colts, national football league, league) — Source: "Michael Allen Doss (born March 24, 1981) is an American former college and professional football player who was a safety in the National Football League (NFL) for six seasons during the early 2000s."

Réponse:
```

**Cible (completion):**
```
Super Bowl 50 Sources: [6][9][11][12]
```

---

## Exemple 4 (qid: 5a8f713d5542995b44242042, aléatoire)

**Question:** You've Got is a daily web series from who, notable guests have included is Joan Alexandra Molinsky (June 8, 1933 – September 4, 2014), widely known as Joan Rivers, was an American comedian, actress, writer, producer, and television host?

**Réponse gold:** AOL

**Triplets dans contexte (5):**
  [1] (edgar rosenberg, joan rivers, spouse)
  [2] (pat sajak weekend, fox news channel, original broadcaster)
  [3] (edgar rosenberg, joan rivers, spouse)
  [4] (jamie kelso, june 8, 1948, date of birth)
  [5] (joan rivers, september 4, 2014, date of death) ✓

**Prompt:**
```
Question: You've Got is a daily web series from who, notable guests have included is Joan Alexandra Molinsky (June 8, 1933 – September 4, 2014), widely known as Joan Rivers, was an American comedian, actress, writer, producer, and television host?

Contexte (triplets de connaissance):
[1] (joan rivers, september 4, 2014, date of death) ✓
[2] (edgar rosenberg, joan rivers, spouse)
[3] (pat sajak weekend, fox news channel, original broadcaster)
[4] (edgar rosenberg, joan rivers, spouse)
[5] (jamie kelso, june 8, 1948, date of birth)

Réponse:
```

**Cible (completion):**
```
AOL Sources: [5]
```

---

## Exemple 5 (qid: 5a8877b6554299206df2b297, aléatoire)

**Question:** Invincible is based on the story of which Philadelphia Eagles player?

**Réponse gold:** Vincent Papale

**Triplets dans contexte (15):**
  [1] (national football conference, american football, sport)
  [2] (national football league, american football, sport)
  [3] (body bag game, november 12, 1990, point in time)
  [4] (philadelphia bell, american football, sport) ✓
  [5] (national football league, american football, sport) ✓
  [6] (linebacker, american football, sport)
  [7] (ken rose (gridiron football), has_info, sport)
  [8] (invincible (2006 film), has_info, member of sports team) ✓
  [9] (vince papale, philadelphia eagles, member of sports team) ✓
  [10] (list of philadelphia eagles head coaches, has_info, sport)
  [11] (philadelphia eagles, american football, sport)
  [12] (vince papale, has_info, sport) ✓
  [13] (national football league, american football, sport)
  [14] (philadelphia eagles cheerleaders, 1948, inception)
  [15] (tommy brasher, american football, sport)

**Prompt:**
```
Question: Invincible is based on the story of which Philadelphia Eagles player?

Contexte (triplets de connaissance):
[1] (philadelphia bell, american football, sport) ✓
[2] (national football league, american football, sport) ✓
[3] (invincible (2006 film), has_info, member of sports team) ✓
[4] (vince papale, philadelphia eagles, member of sports team) ✓
[5] (vince papale, has_info, sport) ✓
[6] (national football conference, american football, sport)
[7] (national football league, american football, sport)
[8] (body bag game, november 12, 1990, point in time)
[9] (linebacker, american football, sport)
[10] (ken rose (gridiron football), has_info, sport)
[11] (list of philadelphia eagles head coaches, has_info, sport)
[12] (philadelphia eagles, american football, sport)
[13] (national football league, american football, sport)
[14] (philadelphia eagles cheerleaders, 1948, inception)
[15] (tommy brasher, american football, sport)

Réponse:
```

**Cible (completion):**
```
Vincent Papale Sources: [4][5][8][9][12]
```

---

## Exemple 6 (qid: 5a7f6de95542995d8a8dde67, cosinus)

**Question:** What TV station did both Kevin James and John Cochran work for at one time in their careers?

**Réponse gold:** CBS

**Triplets dans contexte (5):**
  [1] (john cochran (survivor contestant), has_info, part of series) ✓
  [2] (wtac-tv, 1953, inception)
  [3] (list of news anchors on channel 7 (thailand), has_info, country)
  [4] (kevin can wait, september 19, 2016, start time) ✓
  [5] (george fm, new zealand, country)

**Prompt:**
```
Question: What TV station did both Kevin James and John Cochran work for at one time in their careers?

Contexte (triplets de connaissance):
[1] (john cochran (survivor contestant), has_info, part of series) ✓
[2] (kevin can wait, september 19, 2016, start time) ✓
[3] (wtac-tv, 1953, inception)
[4] (list of news anchors on channel 7 (thailand), has_info, country)
[5] (george fm, new zealand, country)

Réponse:
```

**Cible (completion):**
```
CBS Sources: [1][4]
```

---

## Exemple 7 (qid: 5ac311495542995ef918c0dc, cosinus)

**Question:**  Neila Sathyalingam was an alumna of Kalakshetra in Madras under the tutelage of Srimathi an Indian theosophist and choreographer of what Indian classical dance form?

**Réponse gold:** Bharatanatyam

**Triplets dans contexte (5):**
  [1] (neila sathyalingam, 9 march 2017, date of death) ✓
  [2] (kanaka srinivasan, sangeet natak akademi award, award received)
  [3] (meenakshi chitharanjan, padma shri, award received)
  [4] (andhra pradesh, india, country)
  [5] (thingbaijam babu singh, 8 february 1936, date of birth)

**Prompt:**
```
Question:  Neila Sathyalingam was an alumna of Kalakshetra in Madras under the tutelage of Srimathi an Indian theosophist and choreographer of what Indian classical dance form?

Contexte (triplets de connaissance):
[1] (neila sathyalingam, 9 march 2017, date of death) ✓
[2] (kanaka srinivasan, sangeet natak akademi award, award received)
[3] (meenakshi chitharanjan, padma shri, award received)
[4] (andhra pradesh, india, country)
[5] (thingbaijam babu singh, 8 february 1936, date of birth)

Réponse:
```

**Cible (completion):**
```
Bharatanatyam Sources: [1]
```

---

## Exemple 8 (qid: 5abe3a0955429965af743e9f, aléatoire)

**Question:** Who was the director of the 1998 action-thriller disaster film produced by Mark Gordon for which Graham Yost is best-known?

**Réponse gold:** Mikael Salomon

**Triplets dans contexte (4):**
  [1] (broken arrow, graham yost, screenwriter) ✓
  [2] (armageddon, disaster film, genre)
  [3] (apocalypse (2007 film), has_info, genre)
  [4] (graham yost, september 5, 1959, date of birth) ✓

**Prompt:**
```
Question: Who was the director of the 1998 action-thriller disaster film produced by Mark Gordon for which Graham Yost is best-known?

Contexte (triplets de connaissance):
[1] (broken arrow, graham yost, screenwriter) ✓ — Source: "Graham John Yost (born September 5, 1959) is a Canadian film and television screenwriter."
[2] (graham yost, september 5, 1959, date of birth) ✓
[3] (armageddon, disaster film, genre)
[4] (apocalypse (2007 film), has_info, genre)

Réponse:
```

**Cible (completion):**
```
Mikael Salomon Sources: [1][4]
```

---

## Exemple 9 (qid: 5a8793cb5542996e4f30884e, aléatoire)

**Question:**  W. S. McIntosh pushed for minorities to be able to work at the department store in what Ohio city?

**Réponse gold:** Dayton

**Triplets dans contexte (11):**
  [1] (federated department stores, macy's, subsidiary) ✓
  [2] (market garden brewery, brewery, instance of)
  [3] (stockmann, helsinki, headquarters location)
  [4] (efird's department store, columbia, south carolina, located in administrative territorial entity)
  [5] (stockmann, helsinki, headquarters location)
  [6] (sm store, sm investments, parent organization)
  [7] (rike kumler co., has_info, subsidiary) ✓
  [8] (market garden brewery, brewery, instance of)
  [9] (stockmann, helsinki centre, has_info, headquarters location)
  [10] (great lakes brewing, brewery, instance of)
  [11] (gus blass department store, 1912, inception)

**Prompt:**
```
Question:  W. S. McIntosh pushed for minorities to be able to work at the department store in what Ohio city?

Contexte (triplets de connaissance):
[1] (federated department stores, macy's, subsidiary) ✓
[2] (rike kumler co., has_info, subsidiary) ✓
[3] (market garden brewery, brewery, instance of)
[4] (stockmann, helsinki, headquarters location)
[5] (efird's department store, columbia, south carolina, located in administrative territorial entity)
[6] (stockmann, helsinki, headquarters location)
[7] (sm store, sm investments, parent organization)
[8] (market garden brewery, brewery, instance of)
[9] (stockmann, helsinki centre, has_info, headquarters location)
[10] (great lakes brewing, brewery, instance of)
[11] (gus blass department store, 1912, inception)

Réponse:
```

**Cible (completion):**
```
Dayton Sources: [1][7]
```

---

## Exemple 10 (qid: 5ab6171b554299488d4d9aa8, aléatoire)

**Question:** Clark Duke of "The Office" and "Greek" also starred in a film based on the book "All the Way" written by what author?

**Réponse gold:** Andy Behrens

**Triplets dans contexte (11):**
  [1] (clark duke, may 5, 1985, date of birth) ✓
  [2] (love's enduring promise, hallmark channel, original broadcaster)
  [3] (love comes softly, hallmark channel, original broadcaster)
  [4] (love's christmas journey, has_info, original broadcaster)
  [5] (kick-ass 2 (film), has_info, author)
  [6] (hit-girl, mark millar, author)
  [7] (love's abiding joy, hallmark channel, original broadcaster)
  [8] (clark duke, may 5, 1985, date of birth) ✓
  [9] (love's long journey, hallmark channel, original broadcaster)
  [10] (love's unfolding dream, hallmark channel, original broadcaster)
  [11] (love's unending legacy, hallmark channel, original broadcaster)

**Prompt:**
```
Question: Clark Duke of "The Office" and "Greek" also starred in a film based on the book "All the Way" written by what author?

Contexte (triplets de connaissance):
[1] (clark duke, may 5, 1985, date of birth) ✓
[2] (clark duke, may 5, 1985, date of birth) ✓
[3] (love's enduring promise, hallmark channel, original broadcaster)
[4] (love comes softly, hallmark channel, original broadcaster)
[5] (love's christmas journey, has_info, original broadcaster)
[6] (kick-ass 2 (film), has_info, author)
[7] (hit-girl, mark millar, author)
[8] (love's abiding joy, hallmark channel, original broadcaster)
[9] (love's long journey, hallmark channel, original broadcaster)
[10] (love's unfolding dream, hallmark channel, original broadcaster)
[11] (love's unending legacy, hallmark channel, original broadcaster)

Réponse:
```

**Cible (completion):**
```
Andy Behrens Sources: [1][8]
```

---

