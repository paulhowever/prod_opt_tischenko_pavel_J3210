это лаба 3 по методам оптимизации. суть: своя бинарная нейронка-классификатор, обученная двумя оптимизаторами из прошлых лаб — l-bfgs и наискорейший спуск с армихо. датасеты d1 и d2 лежат в lab3/lab3/, третий датасет d3 закрытый и приходит на защите.

архитектура — обычный mlp: входной полносвязный слой, relu, опциональные residual-блоки, и линейная голова. лосс — bce по логитам через softplus, l2 только на веса (без смещений), pos_weight балансирует классы. бэкпроп — аналитический, проверяется finite-difference в smoke-тестах.

по умолчанию сплит ровно 80/20 train/test как требует тз. опционально через флаг --three-way-split можно включить 64/16/20 с отдельным val для подбора порога.

как запустить
==============

всё работает только из каталога lab3/lab3 (в корне репо нет cargo.toml):

cd lab3/lab3

сборка и тесты (проверка градиентов finite-difference, 3 теста):

cargo test --release

обучение и сохранение модели в json (потом понадобится для d3):

cargo run --release -- train --csv dataset1.csv --method lbfgs --weights model.json --confusion
cargo run --release -- train --csv dataset2.csv --method lbfgs --weights model_d2.json --confusion

сравнение двух оптимизаторов на обоих датасетах + частичная сумма по формуле тз без d3:

cargo run --release -- bench --csv1 dataset1.csv --csv2 dataset2.csv --out-dir bench_out

проверка что метки не зашиты в фичи (обучает на честных и на shuffled-метках, печатает f1, должен быть ok):

cargo run --release -- sanity --csv dataset1.csv --seed 42
cargo run --release -- sanity --csv dataset2.csv --seed 42

опциональные флаги train: --three-way-split, --no-balance, --hidden N, --residual N, --l2 X, --seed N, --out-json report.json, --curve curve.csv. без аргументов программа печатает usage.

что делать когда пришлют d3
============================

кладёшь d3.csv в lab3/lab3 (или указываешь полный путь). модель выбирается по числу признаков feature_* в d3:

если 2 признака — берём model.json (обучен на d1):

cargo run --release -- eval --csv d3.csv --weights model.json --eval-on full --confusion

если 4 признака — берём model_d2.json (обучен на d2):

cargo run --release -- eval --csv d3.csv --weights model_d2.json --eval-on full --confusion

если число признаков другое — придётся переобучать на подходящем датасете (склейка d1+d2 не получится, у них разное число фич: 2 vs 4).

команда eval печатает строку f1 ... — это и есть f1(d3). итоговая оценка считается по формуле 0.3·f1(d1) + 0.3·f1(d2) + 0.4·f1(d3), порог зачёта 0.55. на d1 и d2 модель даёт f1=1.0, так что частичная сумма уже 0.60 и формула гарантированно выше порога даже при f1(d3)=0.

структура
==========

lab1/ — реализация градиентного спуска / ньютона / nelder-mead, базовый трейт objective
lab2/lab2/ — bfgs, l-bfgs, nlcg, армихо line search
lab3/lab3/ — сама лаба 3: mlp, bce, оба оптимизатора через trait objective, cli train/bench/eval/sanity

всё пишется на rust 2021. внешние зависимости минимальные: csv, rand, serde, serde_json, nalgebra.
