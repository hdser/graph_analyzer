WITH 
human_avatars AS (
    SELECT avatar
    FROM public."V_CrcV2_Avatars"
    WHERE "type" = 'CrcV2_RegisterHuman'
)
SELECT 
    LOWER(t1.trustee) AS source,
    LOWER(t1.truster) AS target
FROM "V_CrcV2_TrustRelations" t1
INNER JOIN human_avatars t2 ON t2.avatar = t1.truster
INNER JOIN human_avatars t3 ON t3.avatar = t1.trustee
WHERE t1.truster != t1.trustee
