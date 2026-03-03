SELECT date::DATE AS report_date,
        COUNT(DISTINCT id) AS quantity
FROM schema.table
WHERE date::DATE BETWEEN current_date-3 AND current_date-1
GROUP BY date::DATE
