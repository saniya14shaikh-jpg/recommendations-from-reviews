"""
Dataset Generator — 5000 Amazon-style product reviews
Sentiment: 0 = Negative, 1 = Positive
"""

import csv, random, os
random.seed(42)

PRODUCTS = [
    "wireless headphones","bluetooth speaker","laptop stand","USB hub",
    "mechanical keyboard","gaming mouse","webcam","monitor","phone case",
    "screen protector","power bank","charging cable","smartwatch",
    "fitness tracker","earbuds","tablet","e-reader","smart bulb",
    "coffee maker","air purifier","robot vacuum","instant pot","blender",
    "toaster oven","electric kettle","yoga mat","resistance bands",
    "running shoes","backpack","water bottle","sunglasses","wallet",
]

POS_OPEN = [
    "Absolutely love this {p}!", "This {p} exceeded my expectations.",
    "Best {p} I've ever purchased.", "Amazing quality {p}, highly recommend.",
    "Couldn't be happier with this {p}.", "This {p} is a game changer.",
    "Fantastic {p}, works perfectly.", "Great value for money {p}.",
    "Very impressed with this {p}.", "Outstanding {p}, five stars.",
    "Excellent {p}, exactly as described.", "Super happy with my {p} purchase.",
    "This {p} is worth every penny.", "Solid {p}, highly satisfied.",
    "Purchased this {p} last month and love it.",
]
POS_BODY = [
    "The build quality is top-notch and it arrived quickly.",
    "Setup was easy and it works flawlessly right out of the box.",
    "The performance quality is incredible for the price.",
    "Packaging was great and the product feels premium.",
    "I've been using it daily for weeks and it still works like new.",
    "Battery life is phenomenal — lasts much longer than advertised.",
    "Comfortable to use for long periods without any fatigue.",
    "The design is sleek and modern, looks great on my desk.",
    "Customer support was very responsive when I had questions.",
    "Fits perfectly and matches exactly what was shown in the photos.",
    "My whole family loves it and we've recommended it to friends.",
    "Durable construction — dropped it a few times with no damage.",
    "Connectivity is seamless and lag-free every single time.",
    "Much better than my previous one — noticeable improvement.",
]
POS_CLOSE = [
    "Would definitely buy again!", "Highly recommend to anyone looking for reliability.",
    "5 stars without hesitation.", "Already ordered a second one as a gift.",
    "Will be buying more products from this brand.",
    "Very satisfied, no complaints.", "A must-buy!", "10/10 would recommend!",
]

NEG_OPEN = [
    "Very disappointed with this {p}.", "Do NOT buy this {p}.",
    "Terrible {p}, complete waste of money.", "This {p} stopped working after one week.",
    "Worst {p} I have ever purchased.", "Deeply unsatisfied with this {p}.",
    "This {p} is a total scam.", "Regret buying this {p}.",
    "Poor quality {p}, save your money.", "This {p} broke within days.",
    "Not as described — very misleading {p} listing.",
    "Huge disappointment, expected much better from this {p}.",
    "This {p} is cheaply made and falls apart.",
    "Zero stars if I could for this {p}.", "Faulty {p}, does not work as advertised.",
]
NEG_BODY = [
    "The quality is extremely poor and feels like it will break any day.",
    "It stopped working completely after just a few days of light use.",
    "The packaging was damaged and the product arrived broken.",
    "Took forever to arrive and the item was not as described at all.",
    "Battery dies within an hour — completely unusable for daily tasks.",
    "Very uncomfortable to use and causes strain after just minutes.",
    "The design looks nothing like the photos — very misleading listing.",
    "Customer support never responded to my multiple messages.",
    "Connection drops constantly and the device overheats quickly.",
    "Cheaply made materials that scratched and peeled after first use.",
    "Instructions were incomprehensible and setup took hours.",
    "Performance is terrible compared to competitors at the same price.",
    "Loud rattling noise that makes it completely unusable.",
    "Already broke and I only used it twice — total waste of money.",
]
NEG_CLOSE = [
    "Returning this immediately.", "Do not waste your money.",
    "1 star — would give zero if possible.", "Worst purchase I've made this year.",
    "Requested a refund right away.", "Buyer beware — stay far away.",
    "Very angry and will not buy from this seller again.", "Absolute rubbish.",
]

def make_review(positive: bool):
    p = random.choice(PRODUCTS)
    if positive:
        text = f"{random.choice(POS_OPEN).format(p=p)} {random.choice(POS_BODY)} {random.choice(POS_CLOSE)}"
        rating = random.choice([4,5,5,5])
        label = 1
    else:
        text = f"{random.choice(NEG_OPEN).format(p=p)} {random.choice(NEG_BODY)} {random.choice(NEG_CLOSE)}"
        rating = random.choice([1,1,2])
        label = 0
    return p, text, rating, label

def generate_dataset(n=5000, path="amazon_reviews.csv"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows, n_pos = [], int(n * 0.55)
    users = [f"user_{i:04d}" for i in range(1,1001)]
    verified = ["Verified Purchase","Verified Purchase","Verified Purchase",""]
    for flag, count in [(True, n_pos),(False, n - n_pos)]:
        for _ in range(count):
            product, text, rating, label = make_review(flag)
            rows.append({"username":random.choice(users),"product":product,
                         "rating":rating,"review_text":text,"sentiment":label,
                         "verified":random.choice(verified),"helpful_votes":random.randint(0,150)})
    random.shuffle(rows)
    for i, r in enumerate(rows,1): r["review_id"] = f"REV{i:05d}"
    fields = ["review_id","username","product","rating","review_text","sentiment","verified","helpful_votes"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    pos = sum(1 for r in rows if r["sentiment"]==1)
    print(f"✅ Dataset: {path} | Total:{len(rows)} Positive:{pos} Negative:{len(rows)-pos}")

if __name__ == "__main__":
    generate_dataset(5000, "amazon_reviews.csv")