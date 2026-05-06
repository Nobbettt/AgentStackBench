DIFF_PYTHON = '''
```python
### example/hello.py
<<<<<<< SEARCH
print("Hello World")
=======
print("Hello")
print("World")
>>>>>>> REPLACE
```
'''

DIFF_JAVA = '''
```java
### example/Hello.java
<<<<<<< SEARCH
System.out.println("Hello World");
=======
System.out.println("Hello");
System.out.println("World");
>>>>>>> REPLACE
```
'''

DIFF_GO = '''
```go
### example/hello.go
<<<<<<< SEARCH
fmt.Println("Hello World")
=======
fmt.Println("Hello")
fmt.Println("World")
>>>>>>> REPLACE
```
'''

DIFF_RUST = '''
```rust
### example/hello.rs
<<<<<<< SEARCH
fn main() {
    println!("Hello World");
}
=======
fn main() {
    println!("Hello");
    println!("World");
}
>>>>>>> REPLACE
```
'''

DIFF_CPP = '''
```cpp
### example/hello.cpp
<<<<<<< SEARCH
#include <iostream>
=======
#include <iostream>
>>>>>>> REPLACE

<<<<<<< SEARCH
std::cout << "Hello World";
=======
std::cout << "Hello";
std::cout << "World";
>>>>>>> REPLACE
```
'''

DIFF_C = '''
```c
### example/hello.c
<<<<<<< SEARCH
#include <stdio.h>
=======
#include <stdio.h>
>>>>>>> REPLACE

<<<<<<< SEARCH
printf("Hello World\n");
=======
printf("Hello\n");
printf("World\n");
>>>>>>> REPLACE
```
'''

DIFF_TYPESCRIPT = '''
```typescript
### example/hello.ts
<<<<<<< SEARCH
console.log("Hello World");
=======
console.log("Hello");
console.log("World");
>>>>>>> REPLACE
```
'''

DIFF_JAVASCRIPT = '''
```javascript
### example/hello.js
<<<<<<< SEARCH
console.log("Hello World");
=======
console.log("Hello");
console.log("World");
>>>>>>> REPLACE
```
'''
