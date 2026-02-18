"""
Generate production-quality ASA datasets with genuine difficulty gradient.
Produces: cal_data.json, train_data.json, valid_data.json, test_data.json

Key improvement: ~30% "hard" boundary cases where tool/non-tool
distinction requires semantic understanding, not keyword matching.

Usage: python generate_data.py
"""
import json, random, hashlib
from pathlib import Path

random.seed(42)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

T, N = 1, 0  # tool-necessary, non-tool

# ══════════════════════════════════════════════════════════════════════════
# MATH DOMAIN
# ══════════════════════════════════════════════════════════════════════════
MATH = [
    # ── Tool-necessary: specific computation required ──
    ("Calculate 847 multiplied by 293.", T),
    ("What is 15329 divided by 47?", T),
    ("Find 17.3% of $892.45.", T),
    ("Compute the square root of 7841.", T),
    ("What is 2^31 minus 1?", T),
    ("Convert 3.7 miles to kilometers.", T),
    ("How many seconds are in 3.75 hours?", T),
    ("Calculate the area of a triangle with base 14.7 cm and height 9.3 cm.", T),
    ("What is the volume of a sphere with radius 6.2 meters?", T),
    ("If I invest $5000 at 7.2% compound interest annually for 15 years, what will it be worth?", T),
    ("Calculate the monthly payment on a $320,000 mortgage at 4.5% for 30 years.", T),
    ("What is the standard deviation of 23, 45, 67, 12, 89, 34?", T),
    ("Find the mean and median of 150, 200, 175, 225, 190, 210, 180.", T),
    ("Convert 98.6 degrees Fahrenheit to Celsius.", T),
    ("How much is 3/7 plus 5/11?", T),
    ("What's the hypotenuse of a right triangle with legs 8.5 and 12.3?", T),
    ("Calculate 15 factorial.", T),
    ("What is the circumference of a circle with diameter 23.8 cm?", T),
    ("How many combinations of 5 items from a set of 20?", T),
    ("Find the present value of $10000 received in 8 years at a 6% discount rate.", T),
    ("Calculate the BMI for someone who is 175 cm tall and weighs 82 kg.", T),
    ("What is sin(37 degrees)?", T),
    ("Find the determinant of the matrix [[3,7],[2,5]].", T),
    ("If a car depreciates 15% per year, what is a $28000 car worth after 4 years?", T),
    ("Calculate the tip: the bill is $87.50 and I want to tip 18%.", T),
    ("What percentage is 347 of 1523?", T),
    ("How much interest accrues on $15000 at 3.8% simple interest for 7 months?", T),
    ("Find the GCD of 462 and 1071.", T),
    ("What is 0.0375 expressed as a fraction in lowest terms?", T),
    ("Calculate rent per day if monthly rent is $2350 for a 31-day month.", T),
    ("Total cost: 3 items at $14.99, 2 items at $27.50, plus 8.25% sales tax.", T),
    ("Find the distance between points (3.5, -2.1) and (-4.7, 6.3).", T),
    ("If I drive 340 miles on 12.8 gallons, what's my fuel efficiency in MPG?", T),
    ("What's the compound amount of $7500 at 5.5% quarterly for 6 years?", T),
    ("Convert 2.5 liters per 100km to miles per gallon.", T),
    ("Calculate the Z-score for a value of 78 in a distribution with mean 65 and SD 10.", T),
    ("A recipe calls for 2.5 cups of flour for 12 cookies. How much for 30 cookies?", T),
    ("What is the sum of integers from 1 to 500?", T),
    ("Find the loan balance after 5 years on a $200000 loan at 5% over 30 years.", T),
    ("Calculate the weighted GPA: A(4pts, 3 credits), B+(3.3, 4 credits), A-(3.7, 3 credits).", T),
    # ── Templates for more tool-necessary ──
    ("What is 9473 plus 6821?", T),
    ("Subtract 3847 from 10204.", T),
    ("Multiply 567 by 891.", T),
    ("Divide 98765 by 321.", T),
    ("What is 23.7% of 4580?", T),
    ("Calculate 1.045^20.", T),
    ("Find the cube root of 1728.", T),
    ("How many permutations of 8 items taken 3 at a time?", T),
    ("What is the area of a trapezoid with parallel sides 12 and 18, height 7?", T),
    ("Convert 275 pounds to kilograms.", T),
    ("If electricity costs $0.12 per kWh, how much for running a 1500W heater for 8 hours?", T),
    ("Calculate the diagonal of a rectangle with sides 15 and 20.", T),
    ("What is log base 2 of 1024?", T),
    ("Find the monthly savings needed to reach $50000 in 5 years at 4% annual return.", T),
    ("What is 7/8 minus 3/5?", T),
    ("Calculate the perimeter of a regular hexagon with side length 9.4 cm.", T),
    ("How many tiles of 30cm x 30cm are needed to cover a 4.5m x 3.6m floor?", T),
    ("What's the effective annual rate for 6% compounded monthly?", T),
    ("Convert 1013.25 hectopascals to atmospheres.", T),
    ("Calculate the total surface area of a cylinder with radius 5 and height 12.", T),
    ("What is the correlation coefficient between [1,2,3,4,5] and [2,4,5,4,5]?", T),
    ("If I earn $72,000/year, what's my biweekly gross pay?", T),
    ("Find the roots of x^2 - 7x + 12 = 0.", T),
    ("What is tan(60 degrees)?", T),
    ("Calculate the break-even point: fixed cost $15000, price $25, variable cost $10.", T),
    ("How many days between March 15, 2024 and November 28, 2024?", T),
    ("What is the nth term of 3, 7, 11, 15... for n=50?", T),
    ("Convert speed of sound 343 m/s to km/h.", T),
    ("Calculate the probability of rolling a sum of 7 with two dice.", T),
    ("What is the area under the curve y=x^2 from 0 to 5?", T),

    # ── Non-tool: conceptual, educational, historical ──
    ("Explain why division by zero is undefined.", N),
    ("What is the significance of Euler's number e in mathematics?", N),
    ("How does calculus differ from algebra conceptually?", N),
    ("Why are prime numbers important in cryptography?", N),
    ("Explain the concept of mathematical proof by contradiction.", N),
    ("What is the historical significance of the number zero?", N),
    ("How did Isaac Newton contribute to mathematics?", N),
    ("What makes topology different from geometry?", N),
    ("Explain the difference between discrete and continuous mathematics.", N),
    ("Why is pi irrational, and what does that mean?", N),
    ("What are the practical applications of linear algebra?", N),
    ("How does probability theory relate to everyday decision making?", N),
    ("What is the philosophical debate about whether math is discovered or invented?", N),
    ("Explain the concept of mathematical induction.", N),
    ("What makes the Fibonacci sequence appear in nature?", N),
    ("How do fractals relate to chaos theory?", N),
    ("What is the difference between necessary and sufficient conditions in logic?", N),
    ("Explain why there are different sizes of infinity.", N),
    ("What role does abstract algebra play in modern physics?", N),
    ("How did the development of calculus change science?", N),
    ("What is game theory and how is it applied?", N),
    ("Explain the concept of mathematical limits in simple terms.", N),
    ("What are eigenvalues and why do they matter?", N),
    ("How does Bayesian reasoning differ from frequentist statistics?", N),
    ("What is the significance of Godel's incompleteness theorems?", N),
    ("Why do mathematicians care about the Riemann hypothesis?", N),
    ("Explain the difference between correlation and causation.", N),
    ("What is the intuition behind the central limit theorem?", N),
    ("How is math used in machine learning algorithms?", N),
    ("What advice would you give to someone struggling with math anxiety?", N),

    # ── HARD non-tool: math-sounding but no computation needed ──
    ("What is two plus two?", N),
    ("The square root of nine is what?", N),
    ("Is one hundred a perfect square?", N),
    ("What number comes after ninety-nine?", N),
    ("How many sides does a triangle have?", N),
    ("Is seven a prime number?", N),
    ("What is half of twenty?", N),
    ("How many degrees in a right angle?", N),
    ("What is three squared?", N),
    ("Is zero an even number?", N),

    # ── HARD tool: phrased conversationally, not using "calculate" ──
    ("I'm splitting a $247.83 dinner bill among 7 people. How much does each person owe?", T),
    ("My recipe serves 4 but I'm cooking for 11. It needs 2.75 cups of flour. How much do I need?", T),
    ("I drove 427 miles and used 13.2 gallons. When gas is $3.89/gallon, what was my cost per mile?", T),
    ("I want to tile my 14x18 foot room with tiles that are 0.5x0.5 feet. How many tiles?", T),
    ("My savings account has $12,456 and earns 4.25% APY. What will I have after 3 years?", T),
    ("I need to compare: Product A costs $3.49 for 16oz, Product B costs $5.29 for 28oz. Which is cheaper per ounce?", T),
    ("If I leave at 2:45 PM and drive 183 miles at 65 mph, when do I arrive?", T),
    ("I flew from New York to LA (2475 miles) in 5 hours 23 minutes. Average speed?", T),
    ("My heart rate was 156, 142, 138, 161, 145, 150 during intervals. What's the average?", T),
    ("I'm painting a wall that's 12.5 feet by 9 feet. One gallon covers 350 sq ft. How many gallons?", T),

    # ── More non-tool to balance ──
    ("What is the best way to learn statistics from scratch?", N),
    ("Why do we use base-10 number system?", N),
    ("What career paths are available for mathematicians?", N),
    ("How does number theory connect to computer science?", N),
    ("What is the difference between accuracy and precision?", N),
    ("Explain the monty hall problem in simple terms.", N),
    ("What is a mathematical function, intuitively?", N),
    ("Why is symmetry important in mathematics?", N),
    ("How do you read mathematical notation?", N),
    ("What is the difference between a theorem and a conjecture?", N),
    ("Tell me about the history of algebra.", N),
    ("What is an isomorphism in mathematics?", N),
    ("How do you approach word problems in math?", N),
    ("What is dimensional analysis and why is it useful?", N),
    ("Explain the difference between rational and irrational numbers.", N),
    ("What are some unsolved problems in mathematics?", N),
    ("Why do negative numbers multiplied together give a positive?", N),
    ("What is modular arithmetic used for in daily life?", N),
    ("How did ancient civilizations do mathematics without calculators?", N),
    ("What is the relationship between geometry and art?", N),
]

# ══════════════════════════════════════════════════════════════════════════
# CODE DOMAIN
# ══════════════════════════════════════════════════════════════════════════
CODE = [
    # ── Tool-necessary: code execution ──
    ("Run this Python code: print(sorted([9, 3, 7, 1, 5, 2, 8]))", T),
    ("Execute: import math; print(math.factorial(20))", T),
    ("Run: print([x**2 for x in range(1,11)])", T),
    ("Execute this and show the output: for i in range(5): print(i*i)", T),
    ("Run: import random; random.seed(42); print([random.randint(1,100) for _ in range(5)])", T),
    ("Test this function: def fib(n): return n if n<2 else fib(n-1)+fib(n-2); print(fib(10))", T),
    ("Execute: print('hello world'[::-1])", T),
    ("Run this Python script: d = {'a':1,'b':2,'c':3}; print(sorted(d.items(), key=lambda x: x[1], reverse=True))", T),
    ("Execute: import sys; print(sys.version)", T),
    ("Run: s = 'racecar'; print(s == s[::-1])", T),
    ("Test: print(list(filter(lambda x: x%3==0, range(1,31))))", T),
    ("Execute: print(bin(255))", T),
    ("Run: from collections import Counter; print(Counter('mississippi'))", T),
    ("Execute: print(sum(range(1, 101)))", T),
    ("Run this and check for errors: x = [1,2,3]; x.append([4,5]); print(len(x), x)", T),
    ("Test if this regex works: import re; print(re.findall(r'\\d+', 'abc123def456'))", T),
    ("Execute: print({i: i**3 for i in range(1, 8)})", T),
    ("Run: a = set([1,2,3,4]); b = set([3,4,5,6]); print(a & b, a | b, a - b)", T),
    ("Execute: import json; print(json.dumps({'name':'test','values':[1,2,3]}, indent=2))", T),
    ("Run this to check: list(zip('abcde', range(5)))", T),
    ("Debug this code by running it: def add(a, b): return a + b; print(add('hello', 5))", T),
    ("Execute: print(sorted(['banana','apple','cherry','date'], key=len))", T),
    ("Run: from itertools import permutations; print(list(permutations([1,2,3])))", T),
    ("Execute: matrix = [[1,2],[3,4]]; print([[row[i] for row in matrix] for i in range(2)])", T),
    ("Run and show output: print(*[f'{i}: {chr(i)}' for i in range(65,91)], sep='\\n')", T),
    ("Test: def is_palindrome(s): return s.lower() == s.lower()[::-1]; print(is_palindrome('Racecar'))", T),
    ("Execute: import hashlib; print(hashlib.md5(b'hello').hexdigest())", T),
    ("Run: print('{:,.2f}'.format(1234567.891))", T),
    ("Execute: from datetime import datetime; print(datetime.now().strftime('%Y-%m-%d'))", T),
    ("Run: print(max([3,1,4,1,5,9,2,6,5,3,5], key=lambda x: x))", T),

    # ── More tool-necessary via templates ──
    ("Check what happens when I run: print(type(3.14))", T),
    ("Debug: lst = [1,2,3]; print(lst[5])", T),
    ("Execute: print(all([True, True, False]))", T),
    ("Run this snippet: print(enumerate(['a','b','c']))", T),
    ("Test: print(round(3.14159, 2))", T),
    ("Execute: try: 1/0 \\nexcept ZeroDivisionError as e: print(e)", T),
    ("Run: print(ord('A'), chr(65))", T),
    ("Execute: print(f'{0.123456:.2%}')", T),
    ("Test this comprehension: print({x: x**2 for x in range(6)})", T),
    ("Run: print(bytes(range(10)))", T),
    ("Execute: print(complex(3, 4).conjugate())", T),
    ("Run: print(any(c.isdigit() for c in 'hello123'))", T),
    ("Execute: x = lambda a, b: a if a > b else b; print(x(10, 20))", T),
    ("Run: print(list(map(str, [1,2,3,4,5])))", T),
    ("Test: s = 'Hello World'; print(s.title(), s.swapcase())", T),

    # ── HARD tool: looks educational but actually needs execution ──
    ("Can you verify if this function handles edge cases correctly? def safe_div(a,b): return a/b if b!=0 else None; print(safe_div(10,0), safe_div(10,3))", T),
    ("Will this code work without errors? print(dict(zip(['a','b','c'], [1,2])))", T),
    ("What does this code actually output? x = [i for i in range(10) if i % 2 == 0]; print(x[-1])", T),
    ("Does this list comprehension work? result = [x*y for x in range(3) for y in range(3)]; print(result)", T),
    ("Check if this gives the right answer: print(0.1 + 0.2 == 0.3)", T),
    ("I think there's a bug: d = {}; d.setdefault('key', []).append(1); print(d)", T),
    ("Does Python's sort guarantee stability? Test: print(sorted([(1,'b'),(2,'a'),(1,'a')], key=lambda x: x[0]))", T),
    ("What happens when you do: print([1,2,3] + [4,5] * 2)?", T),
    ("Is this thread-safe? Run and check: import threading; print(threading.active_count())", T),
    ("Verify the output: print(bool(''), bool(' '), bool(0), bool([]), bool([0]))", T),

    # ── Non-tool: concepts, explanations, advice ──
    ("What's the difference between Python lists and tuples?", N),
    ("Explain object-oriented programming to a beginner.", N),
    ("What are the SOLID principles in software engineering?", N),
    ("How does garbage collection work in Python?", N),
    ("What is the difference between compiled and interpreted languages?", N),
    ("Explain the concept of recursion with an analogy.", N),
    ("What are design patterns and why should I learn them?", N),
    ("How does version control with Git work?", N),
    ("What is the difference between a stack and a queue?", N),
    ("Explain the concept of Big-O notation.", N),
    ("What are the pros and cons of dynamically typed languages?", N),
    ("How does memory management differ between C and Python?", N),
    ("What is functional programming?", N),
    ("Explain the concept of API design best practices.", N),
    ("What is the difference between SQL and NoSQL databases?", N),
    ("How do web servers handle multiple requests simultaneously?", N),
    ("What is containerization with Docker?", N),
    ("Explain the difference between concurrency and parallelism.", N),
    ("What are microservices and when should they be used?", N),
    ("How does HTTPS encryption work?", N),
    ("What is test-driven development?", N),
    ("Explain the Model-View-Controller pattern.", N),
    ("What is the difference between REST and GraphQL?", N),
    ("How do you handle technical debt in a codebase?", N),
    ("What are the key principles of clean code?", N),
    ("Explain the CAP theorem in distributed systems.", N),
    ("What is continuous integration and continuous deployment?", N),
    ("How does a hash table work internally?", N),
    ("What is the difference between authentication and authorization?", N),
    ("Best practices for writing maintainable Python code?", N),

    # ── HARD non-tool: code-related but no execution needed ──
    ("What would print('hello') output in Python?", N),
    ("Is 'for x in range(5)' valid Python syntax?", N),
    ("What does the 'self' keyword mean in Python classes?", N),
    ("Does Python use 0-based or 1-based indexing?", N),
    ("What type does range() return in Python 3?", N),
    ("Is Python pass-by-reference or pass-by-value?", N),
    ("What is a decorator in Python conceptually?", N),
    ("How do you typically structure a Python project?", N),
    ("What is the difference between '==' and 'is' in Python?", N),
    ("What does 'if __name__ == \"__main__\"' do in Python?", N),

    # ── More non-tool ──
    ("What programming language should a beginner start with and why?", N),
    ("How has Python evolved over the years?", N),
    ("What is the role of algorithms in computer science?", N),
    ("Explain the difference between frontend and backend development.", N),
    ("What are lambda functions and when are they useful?", N),
    ("How does async/await work conceptually?", N),
    ("What is the GIL in Python and why does it exist?", N),
    ("Explain the concept of dependency injection.", N),
    ("What are generators in Python?", N),
    ("How do you debug code effectively?", N),
]

# ══════════════════════════════════════════════════════════════════════════
# SEARCH DOMAIN
# ══════════════════════════════════════════════════════════════════════════
SEARCH = [
    # ── Tool-necessary: real-time, current, or specific lookups ──
    ("What is the current price of Bitcoin?", T),
    ("Search for the latest SpaceX Starship launch news.", T),
    ("What is today's weather forecast for Seoul?", T),
    ("Find the latest quarterly earnings report for Apple Inc.", T),
    ("What are the current gas prices in California?", T),
    ("Search for recent developments in CRISPR gene editing.", T),
    ("What movies are currently playing in theaters?", T),
    ("Find the latest COVID-19 vaccination statistics worldwide.", T),
    ("What are the trending topics on social media right now?", T),
    ("Search for the most recent Supreme Court rulings.", T),
    ("What is the current exchange rate between USD and EUR?", T),
    ("Find the latest updates on the James Webb Space Telescope observations.", T),
    ("What are the current stock prices of NVIDIA?", T),
    ("Search for recent breakthroughs in quantum computing.", T),
    ("What are the top news stories this week?", T),
    ("Find the schedule for today's Premier League matches.", T),
    ("What is the current population of Tokyo?", T),
    ("Search for recent changes to immigration policies in Canada.", T),
    ("What are the latest AI research papers published this month?", T),
    ("Find current job market statistics for data scientists.", T),
    ("What new features were announced in the latest iPhone?", T),
    ("Search for the most recent climate change report from the IPCC.", T),
    ("What are the current interest rates set by the Federal Reserve?", T),
    ("Find the latest rankings for world universities.", T),
    ("What is the current status of the Mars Perseverance rover?", T),
    ("Search for recent developments in solid-state batteries.", T),
    ("What are the box office numbers for this weekend's movies?", T),
    ("Find the latest cybersecurity threats and vulnerabilities reported.", T),
    ("What is the current wait time for US passport processing?", T),
    ("Search for the most recent FDA drug approvals.", T),

    # ── More tool-necessary via templates ──
    ("What happened in world news today?", T),
    ("Find the latest research on Alzheimer's treatment.", T),
    ("What are the current airline ticket prices from NYC to London?", T),
    ("Search for the latest election polling data.", T),
    ("What's the current unemployment rate in the US?", T),
    ("Find reviews for the newest Samsung Galaxy phone.", T),
    ("What startups received funding this week?", T),
    ("Search for the latest updates on nuclear fusion progress.", T),
    ("What are the current visa requirements for traveling to Japan?", T),
    ("Find the latest recall notices for automobiles.", T),
    ("What is the current S&P 500 index value?", T),
    ("Search for recent advances in autonomous driving.", T),
    ("What are today's top scientific discoveries?", T),
    ("Find the current ranking of the ATP tennis tour.", T),
    ("What new regulations have been proposed for AI technology?", T),

    # ── HARD tool: sounds general but needs current information ──
    ("Who is the current CEO of Twitter?", T),
    ("What is the latest version of Python?", T),
    ("How many subscribers does the most popular YouTube channel have now?", T),
    ("What country currently holds the presidency of the UN Security Council?", T),
    ("Is there a new variant of COVID being tracked?", T),
    ("What is the most downloaded app on the App Store right now?", T),
    ("Who won the most recent Nobel Prize in Physics?", T),
    ("What is the current world record for the 100m sprint?", T),
    ("How much does a Tesla Model 3 cost right now?", T),
    ("What is the current minimum wage in New York?", T),

    # ── Non-tool: general knowledge, historical, conceptual ──
    ("What is the theory of general relativity?", N),
    ("How does photosynthesis work?", N),
    ("What caused World War I?", N),
    ("Explain the water cycle.", N),
    ("What is the difference between DNA and RNA?", N),
    ("How do black holes form?", N),
    ("What is the greenhouse effect?", N),
    ("Explain the process of natural selection.", N),
    ("What are the basic principles of democracy?", N),
    ("How does the human immune system work?", N),
    ("What is the significance of the Rosetta Stone?", N),
    ("Explain how supply and demand affects prices.", N),
    ("What are the main differences between socialism and capitalism?", N),
    ("How do vaccines work?", N),
    ("What is string theory in simple terms?", N),
    ("Explain the concept of entropy.", N),
    ("What is the difference between weather and climate?", N),
    ("How does the electoral college system work?", N),
    ("What are the main causes of the French Revolution?", N),
    ("Explain how nuclear power plants generate electricity.", N),
    ("What is the significance of the Magna Carta?", N),
    ("How do earthquakes occur?", N),
    ("What is the heliocentric model of the solar system?", N),
    ("Explain the concept of cognitive bias.", N),
    ("What are the philosophical ideas of Stoicism?", N),
    ("How does inflation affect an economy?", N),
    ("What is the trolley problem in ethics?", N),
    ("Explain how the internet works at a high level.", N),
    ("What are the stages of human cognitive development?", N),
    ("How does a democratic republic differ from a direct democracy?", N),

    # ── HARD non-tool: sounds like it needs lookup but is common knowledge ──
    ("What is the capital of France?", N),
    ("Who was the first person to walk on the Moon?", N),
    ("What is the chemical formula for water?", N),
    ("How many continents are there on Earth?", N),
    ("Who wrote Romeo and Juliet?", N),
    ("What is the speed of light approximately?", N),
    ("What is the largest planet in our solar system?", N),
    ("Who painted the Mona Lisa?", N),
    ("What year did World War II end?", N),
    ("What is the boiling point of water in Celsius?", N),

    # ── More non-tool ──
    ("What makes the Renaissance an important period in history?", N),
    ("How is artificial intelligence different from human intelligence?", N),
    ("What are the ethical concerns around genetic engineering?", N),
    ("How do birds navigate during migration?", N),
    ("What is the significance of the discovery of penicillin?", N),
    ("Explain the concept of opportunity cost in economics.", N),
    ("What role does the United Nations play in global affairs?", N),
    ("How do languages evolve over time?", N),
    ("What is the social contract theory?", N),
    ("Explain the difference between deductive and inductive reasoning.", N),
]

# ══════════════════════════════════════════════════════════════════════════
# TRANSLATION DOMAIN
# ══════════════════════════════════════════════════════════════════════════
TRANSLATION = [
    # ── Tool-necessary: specific translation requests ──
    ("Translate 'I love programming and building things' to Japanese.", T),
    ("How do you say 'good morning, how are you?' in Korean?", T),
    ("Translate this to French: 'The meeting has been rescheduled to next Tuesday.'", T),
    ("What is 'machine learning' in Mandarin Chinese?", T),
    ("Translate 'Please handle this with care' to German.", T),
    ("How would you say 'I need to book a hotel room' in Spanish?", T),
    ("Translate to Arabic: 'The conference will begin at 9 AM.'", T),
    ("What is 'artificial intelligence' in Russian?", T),
    ("Translate 'Where is the nearest hospital?' to Portuguese.", T),
    ("How do you say 'Thank you for your patience' in Italian?", T),
    ("Translate this to Korean: 'Open source is about sharing knowledge freely.'", T),
    ("What is 'sustainable development' in French?", T),
    ("Translate 'I would like to cancel my reservation' to Japanese.", T),
    ("How do you say 'The weather is beautiful today' in German?", T),
    ("Translate to Chinese: 'Data privacy is a fundamental right.'", T),
    ("What is 'neural network' in Spanish?", T),
    ("Translate 'Could you please speak more slowly?' to French.", T),
    ("How do you say 'The project deadline is approaching' in Korean?", T),
    ("Translate to Russian: 'Education is the key to success.'", T),
    ("What is 'climate change adaptation' in Arabic?", T),
    ("Translate 'I am looking for a software engineering position' to Japanese.", T),
    ("How do you express 'It was a pleasure meeting you' in Mandarin?", T),
    ("Translate this formal text to German: 'Dear Sir/Madam, I am writing to inquire about...'", T),
    ("What is 'renewable energy sources' in Portuguese?", T),
    ("Translate 'The train departs at 3:45 PM from platform 7' to Italian.", T),
    ("How do you say 'I need a receipt please' in Korean?", T),
    ("Translate to French: 'The algorithm processes data in real time.'", T),
    ("What is 'deep learning' in Chinese?", T),
    ("Translate 'This product contains allergens' to Spanish.", T),
    ("How do you say 'Let me know if you have any questions' in Japanese?", T),

    # ── More tool-necessary ──
    ("Translate 'Debugging code is an important skill' to Korean.", T),
    ("What is 'biodiversity conservation' in French?", T),
    ("Translate to German: 'The warranty expires after two years.'", T),
    ("How do you say 'The server is currently under maintenance' in Chinese?", T),
    ("Translate 'Please fasten your seatbelt' to Arabic.", T),
    ("What is 'quantum computing' in Japanese?", T),
    ("Translate to Spanish: 'The results exceeded our expectations.'", T),
    ("How do you express 'I apologize for the inconvenience' in French?", T),
    ("Translate 'Keep refrigerated after opening' to Italian.", T),
    ("What is 'cybersecurity' in Russian?", T),
    ("How do you say 'The presentation will be recorded' in Korean?", T),
    ("Translate to Portuguese: 'Innovation drives economic growth.'", T),
    ("What is 'autonomous vehicle' in German?", T),
    ("Translate 'No refunds after 30 days' to Chinese.", T),
    ("How do you say 'I would like to speak with a manager' in Spanish?", T),

    # ── HARD tool: indirect translation request ──
    ("I'm emailing a colleague in Tokyo. How would I write 'I look forward to our collaboration' in their language?", T),
    ("My French host family doesn't speak English. How do I tell them 'I have a nut allergy' in their language?", T),
    ("I'm traveling to Seoul next week. How would I ask for directions to a subway station in the local language?", T),
    ("I need to label this product for the German market. What does 'Use before date' translate to?", T),
    ("My Brazilian client sent an email. Can you tell me what 'Precisamos agendar uma reuniao' means?", T),
    ("For my Spanish homework, how do I write a formal greeting and introduction?", T),
    ("What would the Chinese equivalent of the English idiom 'kill two birds with one stone' be?", T),
    ("I want to write 'Happy New Year' on cards for my international colleagues in Japanese, Korean, and French.", T),
    ("A Russian document says 'Политика конфиденциальности'. What does that mean?", T),
    ("How would a native Korean speaker express the concept of 'work-life balance'?", T),

    # ── Non-tool: linguistics, language theory, learning advice ──
    ("What are the most widely spoken languages in the world?", N),
    ("How do children naturally acquire language?", N),
    ("What is the Sapir-Whorf hypothesis about language and thought?", N),
    ("Explain the difference between analytic and synthetic languages.", N),
    ("How does language influence culture?", N),
    ("What makes Japanese writing system unique?", N),
    ("How are sign languages structured compared to spoken languages?", N),
    ("What is the best way to learn a second language as an adult?", N),
    ("Explain the concept of linguistic relativity.", N),
    ("What are the challenges of machine translation?", N),
    ("How do tonal languages like Mandarin work?", N),
    ("What is the history of the English language?", N),
    ("How many languages are there in the world?", N),
    ("What causes languages to go extinct?", N),
    ("Explain the difference between pidgin and creole languages.", N),
    ("What is code-switching in bilingual speakers?", N),
    ("How does Korean hangul differ from Chinese characters?", N),
    ("What are the common features shared by all human languages?", N),
    ("How does language processing work in the brain?", N),
    ("What is the difference between translation and interpretation?", N),
    ("How has technology changed the way we learn languages?", N),
    ("What makes some languages harder to learn than others?", N),
    ("Explain the concept of language families and their origins.", N),
    ("What role does grammar play in effective communication?", N),
    ("How do dialects form within a language?", N),
    ("What is the relationship between language and identity?", N),
    ("How do writing systems evolve?", N),
    ("What are endangered languages and why should we preserve them?", N),
    ("How does bilingualism affect cognitive development?", N),
    ("What is computational linguistics?", N),

    # ── HARD non-tool: mentions languages but no translation needed ──
    ("What does 'carpe diem' mean?", N),
    ("What is the origin of the word 'algorithm'?", N),
    ("What does 'c'est la vie' roughly mean?", N),
    ("What language does the word 'tsunami' come from?", N),
    ("What does 'e pluribus unum' mean and where is it used?", N),
    ("What is the meaning of 'schadenfreude'?", N),
    ("Where does the word 'robot' originate from?", N),
    ("What does 'bon appetit' mean?", N),
    ("What language is the phrase 'hakuna matata' from?", N),
    ("What is the etymology of the word 'computer'?", N),

    # ── More non-tool ──
    ("What are the benefits of learning multiple languages?", N),
    ("How accurate are modern translation apps?", N),
    ("What is the future of simultaneous translation technology?", N),
    ("How do translators handle untranslatable words?", N),
    ("What makes poetry particularly difficult to translate?", N),
    ("How does context affect the meaning of words across languages?", N),
    ("What are false friends in language learning?", N),
    ("How has English become a global lingua franca?", N),
    ("What is the role of immersion in language learning?", N),
    ("How does subtitling differ from dubbing in foreign films?", N),
]


def assign_ids_and_split(domain_samples, domain_name):
    """Shuffle, assign unique IDs, and split into CAL/TRAIN/VALID/TEST."""
    tool = [(inst, label) for inst, label in domain_samples if label == T]
    nontool = [(inst, label) for inst, label in domain_samples if label == N]

    random.shuffle(tool)
    random.shuffle(nontool)

    # Need: CAL 40+40, TRAIN 40+40, VALID 20+20, TEST 40+40
    assert len(tool) >= 140, f"{domain_name}: need 140 tool, got {len(tool)}"
    assert len(nontool) >= 140, f"{domain_name}: need 140 non-tool, got {len(nontool)}"

    splits = {}
    split_sizes = [("cal", 40), ("train", 40), ("valid", 20), ("test", 40)]
    t_idx, n_idx = 0, 0

    for split_name, size in split_sizes:
        samples = []
        for i in range(size):
            inst, label = tool[t_idx]
            samples.append({
                "id": f"{domain_name}_{split_name}_t_{i+1:03d}",
                "instruction": inst,
                "domain": domain_name,
                "label": T,
            })
            t_idx += 1
        for i in range(size):
            inst, label = nontool[n_idx]
            samples.append({
                "id": f"{domain_name}_{split_name}_n_{i+1:03d}",
                "instruction": inst,
                "domain": domain_name,
                "label": N,
            })
            n_idx += 1
        splits[split_name] = samples

    return splits


def main():
    all_splits = {"cal": [], "train": [], "valid": [], "test": []}

    for domain_name, domain_data in [
        ("math", MATH), ("code", CODE), ("search", SEARCH), ("translation", TRANSLATION)
    ]:
        t_count = sum(1 for _, l in domain_data if l == T)
        n_count = sum(1 for _, l in domain_data if l == N)
        print(f"  {domain_name}: {t_count} tool + {n_count} non-tool = {t_count + n_count}")

        splits = assign_ids_and_split(domain_data, domain_name)
        for split_name in all_splits:
            all_splits[split_name].extend(splits[split_name])

    # Shuffle each split
    for split_name in all_splits:
        random.shuffle(all_splits[split_name])

    # Verify no instruction overlap across splits
    all_instructions = set()
    for split_name, samples in all_splits.items():
        for s in samples:
            assert s["instruction"] not in all_instructions, \
                f"Duplicate instruction in {split_name}: {s['instruction'][:50]}"
            all_instructions.add(s["instruction"])

    # Write to JSON
    filenames = {
        "cal": "cal_data.json",
        "train": "train_data.json",
        "valid": "valid_data.json",
        "test": "test_data.json",
    }
    for split_name, filename in filenames.items():
        path = DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_splits[split_name], f, indent=2, ensure_ascii=False)
        n = len(all_splits[split_name])
        t = sum(1 for s in all_splits[split_name] if s["label"] == T)
        print(f"  Wrote {path}: {n} samples ({t} tool / {n-t} non-tool)")

    print(f"\n  Total: {sum(len(v) for v in all_splits.values())} samples")
    print("  Done!")


if __name__ == "__main__":
    main()
