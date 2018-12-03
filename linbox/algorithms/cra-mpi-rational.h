/* Copyright (C) 2007 LinBox
 *
 * ========LICENCE========
 * This file is part of the library LinBox.
 *
 * LinBox is free software: you can redistribute it and/or modify
 * it under the terms of the  GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 */

#pragma once

namespace LinBox {
    struct IntegerHash {
        size_t operator()(const Integer& n) const
        {
            size_t hash = 0u;
            auto mpz = n.get_mpz();
            auto size = std::abs(mpz->_mp_size);
            for (auto i = 0; i < size; ++i) {
                hash += mpz->_mp_d[i];
            }
            return hash;
        }
    };

    template <class RatCRABase>
    struct MPIRationalRemainder {
        typedef typename RatCRABase::Domain Domain;
        typedef typename RatCRABase::DomainElement DomainElement;

    protected:
        RatCRABase Builder_;
        Communicator* _communicator = nullptr;
        unsigned int _processCount = 0u;

    public:
        template <class Param>
        MPIRationalRemainder(const Param& b, Communicator* c)
            : Builder_(b)
            , _communicator(c)
            , _processCount(c->size())
        {
        }

        // @todo Missing non-vector version of operator().
        // But should surely be invisible for us.
        // We should just template BlasVector<Givaro::ZRing<Integer>>

        template <class Function, class PrimeIterator>
        BlasVector<Givaro::ZRing<Integer>>& operator()(BlasVector<Givaro::ZRing<Integer>>& num, Integer& den, Function& Iteration,
                                                       PrimeIterator& primeg)
        {
            // Check if just sequential
            if (_communicator == nullptr || _processCount == 1) {
                RationalRemainder<RatCRABase> sequential(Builder_);
                return sequential(num, den, Iteration, primeg);
            }

            if (_communicator->master()) {
                Domain D(*primeg);       // @fixme Why?
                BlasVector<Domain> r(D); // @fixme What's that?
                master_task(Iteration, D, r);
                return Builder_.result(num, den);
            }
            else {
                worker_task(Iteration);

                // Dummy return, to prevent warnings
                return num;
            }
        }

        template <class Function>
        void worker_task(Function& Iteration)
        {
            LinBox::MaskedPrimeIterator<LinBox::IteratorCategories::HeuristicTag> gen(_communicator->rank(), _processCount);
            std::unordered_set<Integer, IntegerHash> usedPrimes;

            BlasVector<Domain> result;

            while (true) {
                // Stop when poison pill is 1
                int poisonPill = 0;
                _communicator->recv(poisonPill, 0);
                if (poisonPill == 1) break;

                // Find the next valid prime
                Integer p = *(++gen);
                while (Builder_.noncoprime(p) || usedPrimes.find(p) != usedPrimes.end()) {
                    p = *(++gen);
                };
                usedPrimes.insert(p);

                // Compute mod p
                Domain D(p);
                Iteration(result, D);

                // Send back the prime and the result to master
                _communicator->send(p, 0);
                _communicator->send(result, 0);
            }
        }

        void compute_stat_comm(int* primes, BlasVector<Domain>& r, int& pp, int& idle_process, int& poison_pills_left)
        {
            idle_process = 0;

            r.resize(r.size() + 1);
            //  receive the beginnin and end of a vector in heapspace
            _communicator->recv(r.begin(), r.end(), MPI_ANY_SOURCE, 0);

            //  determine which process sent answer
            //  and give them a new tag either to continue or to stop
            idle_process = (_communicator->status()).MPI_SOURCE;

            poison_pills_left -= primes[idle_process - 1];

            // send the tag
            _communicator->send(primes[idle_process - 1], idle_process);

            // Store the corresponding prime number
            pp = r[r.size() - 1];

            // Restructure the vector like before without added prime number
            r.resize(r.size() - 1);
        }

        template <class Function>
        void master_task(Function& Iteration, Domain& D, BlasVector<Domain>& r)
        {
            // ----- Init

            int primes[_processCount - 1];
            for (auto i = 1u; i < _processCount; i++) {
                primes[i - 1] = 0;
                _communicator->send(primes[i - 1], i);
            }
            Builder_.initialize(D, Iteration(r, D));

            // ----- Compute

            int poisonPillsLeft = _processCount - 1;
            int pp;
            int idle_process = 0;
            while (poisonPillsLeft > 0) {
                compute_stat_comm(primes, r, pp, idle_process, poisonPillsLeft);

                Domain D(pp);
                Builder_.progress(D, r);

                primes[idle_process - 1] = (Builder_.terminated()) ? 1 : 0;
            }
        }
    };
}
