# Filters and Search Documentation

Graph Analyzer provides powerful filtering capabilities to select nodes based on their properties. This document covers all filtering options, regex patterns, and search functionality.

## Overview

The application supports three types of filtering:

| Filter Type | Properties | Operators | Use Case |
|-------------|------------|-----------|----------|
| Numeric | Metrics, counts | >, <, =, ≥, ≤, ≠ | Filter by metric values |
| String | IDs, text fields | equals, contains, regex | Find specific nodes |
| Array | Multi-value fields | contains, not contains, regex | Search within arrays |

## Numeric Filtering

### Available Operators

| Operator | Symbol | Description | Example |
|----------|--------|-------------|---------|
| Greater than | `>` | Values strictly greater | `pagerank > 0.01` |
| Less than | `<` | Values strictly less | `in_degree < 5` |
| Equal | `=` | Exact match | `community_id = 3` |
| Greater or equal | `≥` | Greater than or equal | `out_degree ≥ 10` |
| Less or equal | `≤` | Less than or equal | `clustering_coefficient ≤ 0.5` |
| Not equal | `≠` | Not matching | `core_number ≠ 1` |

### Examples

**High-influence nodes**:
```
Property: pagerank
Operator: >
Value: 0.001
```

**Low-activity nodes**:
```
Property: out_degree
Operator: <
Value: 3
```

**Specific community**:
```
Property: community_id
Operator: =
Value: 42
```

**Exclude periphery**:
```
Property: core_number
Operator: >
Value: 1
```

---

## String Filtering

### Available Operators

| Operator | Description | Case Sensitive | Example |
|----------|-------------|----------------|---------|
| Equals | Exact match | No | `id = "node123"` |
| Not Equals | Not matching | No | `label ≠ "unknown"` |
| Contains | Substring match | No | `name contains "test"` |
| Regex | Regular expression | Configurable | `id regex "^0x[a-f0-9]+"` |

### Examples

**Find by ID prefix**:
```
Property: avatar
Operator: contains
Value: 0x1234
```

**Find test accounts**:
```
Property: name
Operator: contains
Value: test
```

**Ethereum addresses**:
```
Property: id
Operator: regex
Value: ^0x[a-fA-F0-9]{40}$
```

---

## Array Filtering

Array properties (like lists of connected nodes, tags, etc.) support special filtering.

### Available Operators

| Operator | Description | Example |
|----------|-------------|---------|
| Contains | Any element matches | Tags contain "suspicious" |
| Not Contains | No element matches | Groups not contain "admin" |
| Regex | Any element matches pattern | Addresses match `0x...` |

### Examples

**Find nodes with specific tag**:
```
Property: tags
Operator: contains
Value: verified
```

**Exclude nodes with property**:
```
Property: flags
Operator: not_contains
Value: trusted
```

---

## Regular Expressions

### Syntax

The filter supports JavaScript regular expressions with optional flags.

**Basic syntax**:
```
pattern          # Case-insensitive by default
/pattern/        # With delimiters (allows flags)
/pattern/i       # Explicit case-insensitive
/pattern/g       # Global matching
/pattern/gi      # Multiple flags
```

### Common Patterns

#### Ethereum Addresses
```regex
^0x[a-fA-F0-9]{40}$
```
Matches: `0x742d35Cc6634C0532925a3b844Bc9e7595f234a1`

#### Hex Strings (Any Length)
```regex
^0x[a-fA-F0-9]+$
```
Matches: `0xabc`, `0x123456789abcdef`

#### Numeric IDs
```regex
^\d+$
```
Matches: `123`, `456789`, `1`

#### UUID Format
```regex
^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$
```
Matches: `550e8400-e29b-41d4-a716-446655440000`

#### Email-like Pattern
```regex
^[\w.+-]+@[\w.-]+\.\w+$
```
Matches: `user@example.com`

#### IP Address
```regex
^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$
```
Matches: `192.168.1.1`, `10.0.0.1`

#### Starts With
```regex
^prefix
```
Matches: `prefixABC`, `prefix123`

#### Ends With
```regex
suffix$
```
Matches: `ABCsuffix`, `123suffix`

#### Contains Word
```regex
\bword\b
```
Matches: `a word here` (word boundaries)

### Special Characters

These characters have special meaning and must be escaped with `\`:

| Character | Escaped | Meaning |
|-----------|---------|---------|
| `.` | `\.` | Any character |
| `*` | `\*` | Zero or more |
| `+` | `\+` | One or more |
| `?` | `\?` | Optional |
| `^` | `\^` | Start of string |
| `$` | `\$` | End of string |
| `[` | `\[` | Character class |
| `]` | `\]` | Character class |
| `(` | `\(` | Group |
| `)` | `\)` | Group |
| `{` | `\{` | Quantifier |
| `}` | `\}` | Quantifier |
| `\|` | `\\|` | OR |
| `\` | `\\` | Escape |

### Character Classes

| Pattern | Matches |
|---------|---------|
| `[abc]` | a, b, or c |
| `[^abc]` | Not a, b, or c |
| `[a-z]` | Lowercase letters |
| `[A-Z]` | Uppercase letters |
| `[0-9]` | Digits |
| `[a-zA-Z0-9]` | Alphanumeric |
| `\d` | Digit (same as [0-9]) |
| `\D` | Non-digit |
| `\w` | Word character [a-zA-Z0-9_] |
| `\W` | Non-word character |
| `\s` | Whitespace |
| `\S` | Non-whitespace |

### Quantifiers

| Pattern | Meaning |
|---------|---------|
| `*` | Zero or more |
| `+` | One or more |
| `?` | Zero or one |
| `{n}` | Exactly n |
| `{n,}` | n or more |
| `{n,m}` | Between n and m |

### Regex Examples

**Ethereum addresses starting with specific bytes**:
```regex
^0x1234[a-fA-F0-9]{36}$
```

**Find IDs with specific prefix followed by numbers**:
```regex
^user_\d{4,}$
```
Matches: `user_1234`, `user_99999`

**Case-insensitive search**:
```regex
/test/i
```
Matches: `Test`, `TEST`, `TeSt`

**Multiple patterns (OR)**:
```regex
^(bot|test|spam)_
```
Matches: `bot_123`, `test_account`, `spam_node`

**Exclude pattern (negative lookahead)**:
```regex
^(?!test_).*$
```
Matches anything NOT starting with `test_`

---

## Node Search

### Quick Search

The search box in the toolbar performs ID-based search:

1. **Exact match**: Enter full node ID
2. **Partial match**: Enter partial ID (case-insensitive)
3. **Multiple matches**: All matching nodes highlighted

### Search Features

| Feature | Description |
|---------|-------------|
| Exact ID | Type full node ID to select |
| Partial match | Partial ID matches all containing nodes |
| Auto-zoom | Found nodes are centered in view |
| Highlight | Matched nodes get `.searched` class |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Enter | Execute search |
| Escape | Clear search |

---

## Filter UI

### Location

The filter UI is in the sidebar under "Filter Nodes" section.

### Components

1. **Property Dropdown**: Select metric or property to filter
2. **Operator Dropdown**: Select comparison operator
3. **Value Input**: Enter filter value
4. **Apply Button**: "Select Matches"
5. **Reset Button**: Clear selection

### Property Types

The dropdown groups properties by type:

```
Numeric
├── in_degree
├── out_degree
├── pagerank
├── ...

Arrays
├── tags
├── connected_nodes
├── ...

Text
├── avatar
├── name
├── label
├── ...
```

### Dynamic UI

The input field changes based on property type:
- **Numeric**: Number input with operators >, <, =, etc.
- **String**: Text input with equals/contains/regex
- **Array**: Text input with contains/regex

---

## API Filtering

### Data Explorer Endpoint

```bash
# Get nodes with filter
curl "http://localhost:8000/api/nodes?offset=0&limit=100"
```

### Response Structure

```json
{
  "nodes": [
    {
      "avatar": "0x1234...",
      "in_degree": 10,
      "out_degree": 5,
      "pagerank": 0.001,
      ...
    }
  ],
  "columns": [
    {"name": "avatar", "type": "string"},
    {"name": "in_degree", "type": "number"},
    ...
  ],
  "total": 10000,
  "offset": 0,
  "limit": 100
}
```

### Client-Side Filtering

The Data Explorer page supports client-side filtering:

```javascript
// Filter by search term
DataExplorer.searchTerm = "0x1234";
DataExplorer.applyFilters();

// Sort by column
DataExplorer.sortColumn = "pagerank";
DataExplorer.sortAsc = false;
DataExplorer.applySorting();
```

---

## Combined Filters

### Multiple Conditions (AND)

Apply filters sequentially - selection is refined:

1. First filter: `pagerank > 0.001` → 500 nodes
2. Apply second filter on selection
3. Second filter: `community_id = 5` → 50 nodes

### Selection Operations

| Button | Action |
|--------|--------|
| Select Matches | Add matching nodes to selection |
| Reset Selection | Clear all selections |

### Workflow Example

```
1. Load graph
2. Filter: pagerank > 0.001 → Select high-influence nodes
3. Note: "Selected 500 nodes"
4. Filter: clustering_coefficient < 0.1 → Refine to bridge nodes
5. Note: "Selected 150 nodes"
6. Use selection for analysis
```

---

## Tips and Best Practices

### Performance

1. **Start broad**: Use simple filters first
2. **Numeric first**: Faster than regex
3. **Limit regex complexity**: Avoid backtracking patterns
4. **Use precomputed metrics**: Filter on computed values

### Regex Tips

1. **Anchor patterns**: Use `^` and `$` for exact matching
2. **Case handling**: Use `/pattern/i` for case-insensitive
3. **Test patterns**: Validate regex before applying
4. **Escape special chars**: Remember to escape `.`, `*`, etc.

### Common Issues

| Issue | Solution |
|-------|----------|
| "Invalid regex" | Check for unescaped special characters |
| No matches found | Verify property name and value |
| Slow filtering | Simplify regex or use numeric filters |
| Case mismatch | Use `/pattern/i` flag |

---

## Examples Gallery

### Find Whale Accounts
```
Property: in_degree
Operator: >
Value: 1000
```

### Find Dormant Accounts
```
Property: out_degree
Operator: =
Value: 0
```

### Find Bridge Nodes
```
Property: betweenness_centrality
Operator: >
Value: 0.05
```

### Find Community Outliers
```
Step 1: Filter community_id = N
Step 2: Filter clustering_coefficient < 0.05
```

### Find Addresses by Pattern
```
Property: avatar
Operator: regex
Value: ^0x[0]{10,}
```
(Addresses with many leading zeros)

### Exclude Test Accounts
```
Property: name
Operator: regex
Value: ^(?!test_|bot_|spam_)
```
(Names NOT starting with test_, bot_, or spam_)