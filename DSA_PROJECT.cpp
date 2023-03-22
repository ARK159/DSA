
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random> 
 
using namespace std;

vector<vector<string>> train_dataset, test_dataset;
struct Node {
    string label;
    map<string, Node*> children;
};
vector<vector<string>> read_csv(string filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    return data;
}

double entropy(const vector<string>& classes) {
    map<string, int> freq;
    for (const auto& c : classes) {
        freq[c]++;
    }
    double ent = 0.0;
    for (const auto& [_, count] : freq) {
        double p = static_cast<double>(count) / classes.size();
        ent -= p * log2(p);
    }
    return ent;
}

double information_gain(const vector<vector<string>>& data, int attr_idx) {
    vector<string> classes(data.size());
    for (int i = 0; i < data.size(); ++i) {
        classes[i] = data[i].back();
    }
    double parent_entropy = entropy(classes);
    map<string, vector<vector<string>>> splits;
    for (const auto& record : data) {
        splits[record[attr_idx]].push_back(record);
    }
    double child_entropy = 0.0;
    for (const auto& [_, subset] : splits) {
        double p = static_cast<double>(subset.size()) / data.size();
        vector<string> q;
        for (const auto& r : subset) {
            q.push_back(r.back());
        }
        child_entropy += p * entropy(q);
        
    }
    return parent_entropy - child_entropy;
}

Node* build_tree(const vector<vector<string>>& data,  const vector<int> &attr_indices) {
    Node* root = new Node();
    vector<string> classes(data.size());
    for (int i = 0; i < data.size(); ++i) {
        classes[i] = data[i].back();
    }
    if (entropy(classes) == 0.0) {
        root->label = classes.front();
        return root;
    }
    if (attr_indices.empty()) {
        map<string, int> freq;
        for (const auto& c : classes) {
            freq[c]++;
        }
        root->label = max_element(freq.begin(), freq.end(), [](const auto& p1, const auto& p2) {
            return p1.second < p2.second;
        })->first;
        return root;
    }
    int best_attr_idx = attr_indices.front();
    double best_gain = -1.0;
    for (const auto& i : attr_indices) {
        double gain = information_gain(data, i);
        if (gain > best_gain) {
            best_attr_idx = i;
            best_gain = gain;
        }
    }
    root->label = to_string(best_attr_idx);
    map<string, vector<vector<string>>> splits;
    for (const auto& record : data) {
        splits[record[best_attr_idx]].push_back(record);
    }
    vector<int> remaining_attr_indices;
    for (const auto& i : attr_indices) {
        if (i != best_attr_idx) {
            remaining_attr_indices.push_back(i);
        }
    }
    for (const auto& [attr_val, subset] : splits) {
        Node* child = build_tree(subset, remaining_attr_indices);
        root->children[attr_val] = child;
        }
return root;
}

string classify(const vector<string>& record, Node* root) {
if (root->children.empty()) {
return root->label;
}
string attr_val = record[stoi(root->label)];
if (root->children.find(attr_val) == root->children.end()) {
return root->label;
}

return classify(record, root->children[attr_val]);
}

vector<vector<string>> bagging(const vector<vector<string>>& data, int num_samples) {
vector<vector<string>> subset(num_samples);
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dis(0, data.size() - 1);
for (auto& record : subset) {
record = data[dis(gen)];
}
return subset;
}

vector<Node*> build_forest(const vector<vector<string>>& data, int num_trees, int num_features) {
vector<Node*> forest(num_trees);
vector<int> attr_indices(data.front().size() - 1);
iota(attr_indices.begin(), attr_indices.end(), 0);
for (int i = 0; i < num_trees; ++i) {
vector<vector<string>> subset = bagging(data, data.size());
random_shuffle(attr_indices.begin(), attr_indices.end());
attr_indices.resize(num_features);
forest[i] = build_tree(subset, attr_indices);
}
return forest;
}

string classify_forest(const vector<string>& record, const vector<Node*>& forest) {
map<string, int> freq;
for (const auto& tree : forest) {
string prediction = classify(record, tree);
freq[prediction]++;
}
return max_element(freq.begin(), freq.end(), [](const auto& p1, const auto& p2) {
return p1.second < p2.second;
})->first;
}

vector<double> test_classifier(const vector<vector<string>>& data, const vector<Node*>& forest,int num_train) {
int num_correct = 0;
int count =0;
for(const auto& record : data) {
string prediction = classify_forest(record, forest);
 
cout<<test_dataset[count][2].substr(1,test_dataset[count][2].size()-2)<<"--->";
if (prediction == record.back()) {  
    num_correct++;
}
cout<<"Predicted"<<"="<<prediction<<"   "<<"Survived"<<"="<<record.back()<<"\n";
count++;
}
return {(double)num_correct,(double)data.size()};
}
void free_tree(Node* root) {
    for (auto& child : root->children) {
        free_tree(child.second);
    }
    delete root;
}

 int main() {
ios_base::sync_with_stdio(false);
cout.tie(NULL);
freopen("output.txt","w",stdout);

 
vector<vector<string>> data=read_csv("titanic.csv");
vector<vector<string>> train_set, test_set;

 
random_device rd;
mt19937 gen(rd());
int num_train = data.size() * 0.7;
train_set.assign(data.begin(), data.begin() + num_train);
test_set.assign(data.begin() + num_train, data.end());



auto dataset=read_csv("train.csv");
num_train = dataset.size() * 0.7;
train_dataset.assign(dataset.begin(), dataset.begin() + num_train);
test_dataset.assign(dataset.begin() + num_train, dataset.end());

 

vector<Node*> forest = build_forest(train_set, 3,3);
auto metrics = test_classifier(test_set, forest,num_train);
cout<<"Train Size--->"<<train_set.size()<<"\n"<<"Test Size--->"<<test_set.size()<<"\n";
cout <<"Correct Predictions---->"<<metrics.front()<<"\n"<<"Data Size---->"<<metrics.back()<<"\n"<<"Accuracy--->"<<(metrics.front()/metrics.back());  
for (const auto& tree : forest) {
    free_tree(tree);
}

 return 0;
 }
 