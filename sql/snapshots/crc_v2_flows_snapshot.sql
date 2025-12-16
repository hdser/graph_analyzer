SELECT 
    LOWER("from") AS source,
    LOWER("to") AS target,
    "timestamp",
    SUM(amount) / POWER(10, 18) AS amount
FROM "CrcV2_StreamCompleted" 
WHERE 
    "blockNumber" <= {block_number}
    AND "from" != "to"
GROUP BY 1, 2, 3