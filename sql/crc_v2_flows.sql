WITH 
human_avatars AS (
  SELECT DISTINCT avatar
  FROM public."V_CrcV2_Avatars"
  WHERE "type" = 'CrcV2_RegisterHuman'
)

SELECT 
    t1."from" AS source
    ,t1."to" AS target
    ,t1.timestamp
    ,SUM(t1.amount)/POWER(10,18) AS amount
FROM "CrcV2_StreamCompleted" t1
INNER JOIN human_avatars t2
    ON t2.avatar = t1. "from"
INNER JOIN human_avatars t3
    ON t3.avatar = t1. "to"
WHERE t1."from" != t1."to"
GROUP BY 1, 2, 3