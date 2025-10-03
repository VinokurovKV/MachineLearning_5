#1
def hello(s=None):
    if s is None or s == "":
      return "Hello!"
    else:
      return f"Hello, {s}!"

#2   
def int_to_roman(x):
  if x > 3999 or x < 1:
    return "Wrong number"

  val_to_roman = [
      (1000, "M"),
      (900, "CM"),
      (500, "D"),
      (400, "CD"),
      (100, "C"),
      (90, "XC"),
      (50, "L"),
      (40, "XL"),
      (10, "X"),
      (9, "IX"),
      (5, "V"),
      (4, "IV"),
      (1, "I")
  ]

  res = []

  for val, roman in val_to_roman:
    count = x // val
    if count > 0:
      res.append(roman * count)
      x -= val * count

  return ''.join(res)

#3
def longest_common_prefix(x):
  if not x:
    return ""

  cleaned_strings = [s.lstrip() for s in x]

  if any(not s for s in cleaned_strings):
    return ""

  first_string = cleaned_strings[0]

  for i in range(len(first_string)):
    current_char = first_string[i]

    for string in cleaned_strings[1:]:
      if i >= len(string) or string[i] != current_char:
        return first_string[:i]

  return first_string
  
#4
class BankCard:
    def __init__(self, total_sum, balance_limit=None):
        self.total_sum = total_sum
        self._initial_balance_limit = balance_limit
        self._balance_calls = 0

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            print(f"Not enough money to spend {sum_spent} dollars.")
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")
        else:
            self.total_sum -= sum_spent
            print(f"You spent {sum_spent} dollars.")

    def __str__(self):
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self._initial_balance_limit is not None:
            if self._balance_calls >= self._initial_balance_limit:
                print("Balance check limits exceeded.")
                raise ValueError("Balance check limits exceeded.")
            self._balance_calls += 1
        return self.total_sum

    @property
    def balance_limit(self):
        if self._initial_balance_limit is not None:
            return self._initial_balance_limit - self._balance_calls
        return None

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, other):
        if not isinstance(other, BankCard):
            raise TypeError("Operands must be of type BankCard.")
        new_limit = max(self._initial_balance_limit or 0, other._initial_balance_limit or 0) or None
        return BankCard(self.total_sum + other.total_sum, new_limit)


    
a = BankCard(10, 2)
print(a.balance) # 10
print(a.balance_limit) # 1
a(5) # You spent 5 dollars.
print(a.total_sum) # 5
print(a) # To learn the balance call balance.
print(a.balance) # 5
try:
  a(6) # Not enough money to spend 6 dollars.
except ValueError:
  pass
a(5) # You spent 5 dollars.
try:
  a.balance # Balance check limits exceeded.
except ValueError:
  pass
a.put(2) # You put 2 dollars.
print(a.total_sum) # 2

#5
def primes():
    yield 2

    n = 3
    while True:
      is_prime = True
      d = 3

      while d * d <= n:
        if n % d == 0:
          is_prime = False
          break
        d += 2

      if is_prime:
        yield n

      n += 2