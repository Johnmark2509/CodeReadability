import subprocess, tempfile, os, re, joblib

# Load the ML model and vectorizer
vectorizer, model = joblib.load('bug_model.pkl')

# Fallback rules
FALLBACK = {
    "';' expected": "Add the missing semicolon at the end of the statement.",
    "cannot find symbol": "Check for typos or missing imports or misspelled identifiers.",
    "missing return type": "Declare a return type (e.g., void) before the method name.",
    "class, interface, or enum expected": "Ensure your braces and class/interface declaration are correct.",
    "illegal start of type": "Remove stray characters or move the code inside a method or class body.",
    "cannot resolve symbol": "Verify that the variable or method name is correct and imported.",
    "variable might not have been initialized": "Ensure the variable is assigned a value on all code paths before use.",
    "incomparable types": "Use equals() or ensure both sides of == are primitive types.",
    "array required but": "You tried to index a non-array—change the variable to an array or use a different type.",
    "incompatible types": "Cast or change the type so that the assignment matches.",
    "reached end of file while parsing": "Add the missing closing brace '}'.",
    "package does not exist": "Check your import statement or package declaration for typos.",
    "method does not override or implement a method from a supertype": "Remove @Override or fix the signature.",
    "enum constant must be an identifier": "Rename your enum constant so it’s a valid Java identifier.",
    "array dimension missing": "Specify the array size like new Type[size] or correct your brackets.",
    "illegal character": "Remove or replace the invalid character—verify your file encoding is UTF-8.",
    "duplicate class": "Ensure only one public class per file and the filename matches the class name.",
    "modifier not allowed here": "Use modifiers (public, static, etc.) only in valid locations.",
    "cannot inherit from final": "You tried to extend a final class—remove the extends or change the class.",
    "<identifier> expected": "Use a valid identifier (letters, digits, _, $), not a keyword.",
    "unexpected type": "Check you’re not using an array as a variable type incorrectly.",
    "missing method body": "Add method implementation or mark the method abstract.",
    "variable .* is already defined": "Rename the duplicate variable in the same scope.",
    "catch or finally expected": "Add a catch or finally block after the try.",
    "annotation type required": "Define annotations using @interface, not @class or @interface.",
    "missing return statement": "Ensure all code paths return a value when method has non-void return.",
    "unclosed string literal": "Terminate your string literal with a closing quote.",
    "octal escape sequence": "Fix your escape sequence; Java uses \\uXXXX or \\n for newlines.",
    "')' expected": "Add the missing closing parenthesis in your expression.",
    "illegal escape character": "Use a valid escape sequence like \\n, \\t, or Unicode \\uXXXX.",
    "reached end of file in comment": "Close your comment (*/) before the end of file.",
    "invalid method declaration; return type required": "Add a return type to your method declaration.",
    "unreachable statement": "Remove or refactor code after return/break that can never be run.",
    "missing package statement": "If you’re using packages, add package your.pkg.name; at top.",
    "bad operand type for unary operator": "Unary operators (like !) require boolean operands.",
    "bad operand type for binary operator": "Check that the operator is used on compatible types.",
    "bad source value": "Use a valid source version for -source (e.g., 1.8, 11).",
    "bootstrap class path not set": "Specify -bootclasspath when cross‑compiling to older Java versions.",
    "already defined in class": "Rename or remove one of the duplicate method definitions so each signature is unique.",
    "else without if": "Add or move the closing brace ‘}’ before the else to match the if."

}

def estimate_time_complexity(code: str) -> str:
    max_depth = 0
    stack = []
    for ln in code.splitlines():
        if re.search(r'\b(for|while)\b', ln):
            stack.append('{')
            max_depth = max(max_depth, len(stack))
        if '}' in ln and stack:
            stack.pop()
    if max_depth >= 3:
        return 'O(n^3)'
    if max_depth == 2:
        return 'O(n^2)'
    if max_depth == 1:
        return 'O(n)'
    return 'O(1)'

def estimate_space_complexity(code: str) -> str:
    allocs = len(re.findall(r'\bnew\b', code))
    return 'O(n)' if allocs > 1 else 'O(1)'

def analyze_bugs(code: str) -> dict:
    original = code.splitlines()
    errors = []
    patched = original.copy()
    seen = set()

    for _ in range(10):
        with tempfile.NamedTemporaryFile(suffix='.java', delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write("\n".join(patched))
            fname = tmp.name

        try:
            proc = subprocess.Popen(['javac', '-Xmaxerrs', '100', fname],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate()
        except FileNotFoundError:
            os.remove(fname)
            return {
                'errors': [],
                'message': 'javac not found.',
                'suggestion': 'Install JDK.',
                'time_complexity': '-',
                'space_complexity': '-'
            }

        os.remove(fname)
        err_str = stderr.decode().strip()
        if not err_str:
            break

        m = re.search(r'^(.*?\.java):(\d+): (.+)$', err_str, re.MULTILINE)
        if not m:
            break

        line_no = int(m.group(2))
        msg = m.group(3).strip()
        if (line_no, msg) in seen:
            break
        seen.add((line_no, msg))

        suggestion = next((fix for pat, fix in FALLBACK.items() if pat.lower() in msg.lower()), None)
        if not suggestion:
            try:
                X_vec = vectorizer.transform([msg])
                suggestion = model.predict(X_vec)[0]
            except Exception:
                suggestion = "No suggestion available."
                
        explanation = "Matched fallback rule" if suggestion in FALLBACK.values() else "Predicted by ML model"

        errors.append({
            'line': line_no,
            'message': msg,
            'suggestion': suggestion,
            'explanation' : explanation
        })

        if "';' expected" in msg:
            idx = line_no - 1
            if idx < len(patched) and not patched[idx].strip().endswith(";"):
                patched[idx] += ";"
            continue

        break

    return {
        'errors': errors,
        'time_complexity': estimate_time_complexity(code),
        'space_complexity': estimate_space_complexity(code)
    }
