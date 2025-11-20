WITH human_avatars AS (
    SELECT "user" AS avatar 
    FROM public."V_CrcV1_Avatars"
    WHERE "type" = 'CrcV1_Signup'
)
SELECT 
    LOWER(t1."user")      AS source,
    LOWER(t1."canSendTo") AS target,
    'v1'::text AS graph_version
FROM "V_CrcV1_TrustRelations" t1
INNER JOIN human_avatars t2 ON t2.avatar = t1."canSendTo"
INNER JOIN human_avatars t3 ON t3.avatar = t1."user"
WHERE LOWER(t1."user") != LOWER(t1."canSendTo")
