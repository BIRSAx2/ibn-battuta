use crate::algorithms::{Solution, TspSolver};
use crate::parser::Tsp;
use rand::prelude::*;
use std::f64;

pub struct GeneticAlgorithm {
    tsp: Tsp,
    population: Vec<Vec<usize>>,
    best_tour: Vec<usize>,
    best_cost: f64,
    population_size: usize,
    ga_choose_ratio: f64,
    mutate_ratio: f64,
    max_generations: usize,
}

impl GeneticAlgorithm {
    pub fn with_options(
        tsp: Tsp,
        population_size: usize,
        ga_choose_ratio: f64,
        mutate_ratio: f64,
        max_generations: usize,
    ) -> GeneticAlgorithm {
        let population_size = population_size.max(2);  // Ensure at least 2 individuals

        let mut ga = GeneticAlgorithm {
            tsp,
            population: Vec::with_capacity(population_size),
            best_tour: vec![],
            best_cost: f64::INFINITY,
            population_size,
            ga_choose_ratio,
            mutate_ratio,
            max_generations,
        };
        ga.initialize_population();
        ga
    }

    fn initialize_population(&mut self) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.population_size {
            let mut tour: Vec<usize> = (0..self.tsp.dim()).collect();
            tour.shuffle(&mut rng);
            self.population.push(tour);
        }
        self.greedy_init();
    }

    fn greedy_init(&mut self) {
        let num_city = self.tsp.dim();
        let mut start_index = 0;
        while self.population.len() < self.population_size {
            let mut rest: Vec<usize> = (0..num_city).collect();
            if start_index >= num_city {
                start_index = rand::thread_rng().gen_range(0..num_city);
                self.population.push(self.population[start_index].clone());
                continue;
            }
            let mut current = start_index;
            rest.retain(|&x| x != current);
            let mut result_one = vec![current];
            while !rest.is_empty() {
                let (tmp_choose, _) = rest.iter()
                    .map(|&x| (x, self.cost(current, x)))
                    .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
                    .unwrap();
                current = tmp_choose;
                result_one.push(tmp_choose);
                rest.retain(|&x| x != tmp_choose);
            }
            self.population.push(result_one);
            start_index += 1;
        }
    }

    fn calculate_fitness(&self, tour: &[usize]) -> f64 {
        1.0 / self.calculate_tour_cost(tour)
    }

    fn ga_parent(&self, scores: &[f64]) -> (Vec<Vec<usize>>, Vec<f64>) {
        let mut indices: Vec<usize> = (0..self.population.len()).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let num_parents = (self.ga_choose_ratio * self.population.len() as f64) as usize;
        let parents: Vec<Vec<usize>> = indices.iter().take(num_parents)
            .map(|&i| self.population[i].clone())
            .collect();
        let parents_score: Vec<f64> = indices.iter().take(num_parents)
            .map(|&i| scores[i])
            .collect();
        (parents, parents_score)
    }

    fn ga_choose<'a>(&self, genes_score: &[f64], genes_choose: &'a [Vec<usize>]) -> (&'a Vec<usize>, &'a Vec<usize>) {
        let sum_score: f64 = genes_score.iter().sum();
        let score_ratio: Vec<f64> = genes_score.iter().map(|&s| s / sum_score).collect();
        let mut rng = rand::thread_rng();
        let mut choose = |ratio: &[f64]| {
            let mut rand = rng.gen::<f64>();
            ratio.iter().position(|&r| {
                rand -= r;
                rand < 0.0
            }).unwrap()
        };
        let index1 = choose(&score_ratio);
        let index2 = choose(&score_ratio);
        (&genes_choose[index1], &genes_choose[index2])
    }

    fn ga_cross(&self, x: &[usize], y: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let len = x.len();
        let mut rng = rand::thread_rng();
        let mut order: Vec<usize> = (0..len).collect();
        order.shuffle(&mut rng);
        let (start, end) = (order[0].min(order[1]), order[0].max(order[1]));

        let mut new_x = vec![usize::MAX; len];
        let mut new_y = vec![usize::MAX; len];
        new_x[start..=end].copy_from_slice(&y[start..=end]);
        new_y[start..=end].copy_from_slice(&x[start..=end]);

        let fill = |new: &mut Vec<usize>, old: &[usize]| {
            let mut j = (end + 1) % len;
            for &city in old.iter().chain(old.iter()) {
                if !new[start..=end].contains(&city) {
                    new[j] = city;
                    j = (j + 1) % len;
                    if j == start {
                        break;
                    }
                }
            }
        };

        fill(&mut new_x, x);
        fill(&mut new_y, y);

        (new_x, new_y)
    }

    fn ga_mutate(&self, gene: &mut Vec<usize>) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.mutate_ratio {
            let len = gene.len();
            let (start, end) = (rng.gen_range(0..len), rng.gen_range(0..len));
            gene[start.min(end)..=start.max(end)].reverse();
        }
    }

    fn ga(&mut self) -> (Vec<usize>, f64) {
        let scores: Vec<f64> = self.population.iter()
            .map(|tour| self.calculate_fitness(tour))
            .collect();

        let (parents, parents_score) = self.ga_parent(&scores);
        let mut tmp_best_one = parents[0].clone();
        let mut tmp_best_score = parents_score[0];

        let mut fruits = parents.clone();
        while fruits.len() < self.population_size {
            let (gene_x, gene_y) = self.ga_choose(&parents_score, &parents);
            let (mut gene_x_new, mut gene_y_new) = self.ga_cross(gene_x, gene_y);

            self.ga_mutate(&mut gene_x_new);
            self.ga_mutate(&mut gene_y_new);

            let x_adp = self.calculate_fitness(&gene_x_new);
            let y_adp = self.calculate_fitness(&gene_y_new);

            if x_adp > y_adp && !fruits.contains(&gene_x_new) {
                fruits.push(gene_x_new);
            } else if !fruits.contains(&gene_y_new) {
                fruits.push(gene_y_new);
            }
        }

        self.population = fruits;

        (tmp_best_one, tmp_best_score)
    }

    fn calculate_tour_cost(&self, tour: &[usize]) -> f64 {
        tour.windows(2)
            .map(|w| self.cost(w[0], w[1]))
            .sum::<f64>()
            + self.cost(*tour.last().unwrap(), tour[0])
    }
}

impl TspSolver for GeneticAlgorithm {
    fn solve(&mut self) -> Solution {
        let mut best_list = Vec::new();
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.max_generations {
            let (tmp_best_one, tmp_best_score) = self.ga();
            if tmp_best_score > best_score {
                best_score = tmp_best_score;
                best_list = tmp_best_one;
            }
        }

        let best_cost = 1.0 / best_score;
        Solution {
            tour: best_list,
            total: best_cost,
        }
    }

    fn tour(&self) -> Vec<usize> {
        self.best_tour.clone()
    }

    fn cost(&self, from: usize, to: usize) -> f64 {
        self.tsp.weight(from, to)
    }

    fn format_name(&self) -> String {
        format!("Genetic Algorithm")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TspBuilder;

    #[test]
    fn test_example() {
        let data = "
        NAME : example
        COMMENT : this is
        COMMENT : a simple example
        TYPE : TSP
        DIMENSION : 5
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
          1 1.2 3.4
          2 5.6 7.8
          3 3.4 5.6
          4 9.0 1.2
          5 6.0 2.2
        EOF
        ";
        let tsp = TspBuilder::parse_str(data).unwrap();
        let dim = tsp.dim();

        let mut solver = GeneticAlgorithm::with_options(tsp, 25, 0.2, 0.05, 500);
        let solution = solver.solve();

        println!("Example solution: {:?}", solution);
        assert_eq!(solution.tour.len(), dim);
    }

    #[test]
    fn test_gr17() {
        let path = "data/tsplib/gr17.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        run_on_instance(tsp);
    }

    #[test]
    fn test_berlin52() {
        let path = "data/tsplib/st70.tsp";
        let tsp = TspBuilder::parse_path(path).unwrap();

        run_on_instance(tsp);
    }

    fn run_on_instance(tsp: Tsp) {
        let m = tsp.dim();

        let mut solver = GeneticAlgorithm::with_options(tsp, 25, 0.2, 0.05, 500);
        let solution = solver.solve();

        println!("Solution: {:?}", solution);
        assert_eq!(solution.tour.len(), m);
    }
}