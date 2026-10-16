CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
INSERT INTO customers VALUES (1,'us','US'),(2,'uk','UK'),(3,'nul',NULL),(4,'de','DE'),(5,'fr','FR');
INSERT INTO orders VALUES (1,2,10),(2,2,NULL),(3,3,NULL),(4,5,2.5),(5,5,7.5);
SELECT c.id, c.name, COALESCE(SUM(o.amount), 0) AS total_order_amount FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE c.country IS NULL OR c.country != 'US' GROUP BY c.id, c.name ORDER BY c.id;
SELECT '---notin';
CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER);
INSERT INTO employees VALUES (1,'ceo',NULL),(2,'mgr',1),(3,'ic1',2),(4,'ic2',2);
SELECT name FROM employees WHERE id NOT IN (SELECT manager_id FROM employees WHERE manager_id IS NOT NULL) ORDER BY name;
