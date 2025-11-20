WITH
human_avatars_v2 AS (
    SELECT avatar 
    FROM public."V_Crc_Avatars"
    WHERE "version" = 2
)
SELECT
    LOWER(r.inviter)                         AS source,
    LOWER(r.avatar)                          AS target
FROM "CrcV2_RegisterHuman" r
INNER JOIN human_avatars_v2 h2 ON LOWER(h2.avatar)=LOWER(r.avatar)  
WHERE r.inviter <> '0x0000000000000000000000000000000000000000'
AND r.inviter <> r.avatar