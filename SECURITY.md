# Security policy

Report vulnerabilities privately through GitHub's private vulnerability reporting on this repository: Security tab, "Report a vulnerability". Do not open a public issue.

Include the package version, the `tokenizer.json` that triggers the problem (or its repository, revision and SHA-256), and a minimal reproducer. We aim to acknowledge reports within 5 working days. This is a goal, not a commercial SLA.

Of particular interest: crafted `tokenizer.json` files that cause excessive memory or CPU use at load or encode time, crashes or unbounded recursion in the loader, input text that defeats the surrogate checks, and anything that makes the library touch the network or the file system beyond the path it was given.
