Load CSV Data from a file and make predictions using the data inside

To run: `python load_csv.py`

Output: 

```
sex                 : [b'male' b'male' b'male' b'male' b'male']
age                 : [23. 25. 28. 28.  4.]
n_siblings_spouses  : [0 0 0 0 3]
parch               : [0 0 0 0 2]
fare                : [13.     7.05   7.896  7.229 27.9  ]
class               : [b'Second' b'Third' b'Third' b'Third' b'Third']
deck                : [b'unknown' b'unknown' b'unknown' b'unknown' b'unknown']
embark_town         : [b'Southampton' b'Southampton' b'Southampton' b'Cherbourg' b'Southampton']
alone               : [b'y' b'y' b'y' b'y' b'n']


age                 : [10. 28. 28. 28. 36.]
n_siblings_spouses  : [3 0 0 0 1]
class               : [b'Third' b'Third' b'Third' b'First' b'Second']
deck                : [b'unknown' b'unknown' b'unknown' b'C' b'unknown']
alone               : [b'n' b'y' b'y' b'y' b'n'] ```



The second is an example with some columns omitted by creating a list of just the columns you intend to use. 

```
SELECT_COLUMNS = ['survived', 'age',
                  'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)

show_batch(temp_dataset)
```
